"""Validation-only tuning. Test data are never evaluated in this script."""
from pathlib import Path
import sys

if str(Path(__file__).resolve().parents[2]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from experiments.dmcrec.main import (make_parser, parse_args, prepare_runtime, load_data,
                                     dmcrec_params, build_baseline, jsonable, write_json)


def tune(args):
    import json
    if args.dry_run:
        print(json.dumps(jsonable(vars(args)), ensure_ascii=False, indent=2))
        return []
    import gc
    import hashlib
    import time
    import numpy as np
    import optuna
    import torch
    from experiments.dmcrec.my_models.myrec import MyRec
    from experiments.dmcrec.exutils import seed_everything, evaluate_recommendations

    environment = prepare_runtime(args)
    settings = ["original", "sparse"] if args.setting == "both" else [args.setting]
    reports = []
    name = args.models[0]
    for setting in settings:
        split, visual_mean = load_data(args, setting)
        if split.val_set is None:
            raise ValueError("Tuning requires an independent validation set")
        identity = jsonable({"options": {k: v for k, v in vars(args).items() if k not in {"trials", "resume", "dry_run", "quiet", "output_dir"}},
                            "data": split.preprocessing_report, "environment": environment})
        digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
        directory = args.output_dir / "tuning" / args.dataset / setting / name / digest
        directory.mkdir(parents=True, exist_ok=True)
        if getattr(args, "search_sampler", "tpe") == "grid":
            import math
            grid = dict(args.search_space)
            if args.tune_inference == "counterfactual":
                grid["cf_reference"] = args.cf_references
            if not grid or args.trials != math.prod(len(values) for values in grid.values()):
                raise ValueError("Grid tuning requires an explicit search space and exactly one trial per grid point")
            sampler = optuna.samplers.GridSampler(grid, seed=args.split_seed)
        else:
            sampler = optuna.samplers.TPESampler(seed=args.split_seed)
        study = optuna.create_study(study_name=f"{name}_{args.dataset}_{setting}_{digest}",
                                    storage="sqlite:///" + str((directory / "study.db").resolve()),
                                    direction="maximize", load_if_exists=args.resume,
                                    sampler=sampler)
        if study.user_attrs.get("identity", identity) != identity:
            raise ValueError("Study identity differs from current data/configuration")
        study.set_user_attr("identity", identity)
        study.set_user_attr("selection_split", "validation")
        write_json(directory / "manifest.json", identity)

        def objective(trial):
            sampled = {}
            if args.search_space:
                sampled = {key: trial.suggest_categorical(key, values) for key, values in args.search_space.items()}
            elif args.params_group == 0:
                sampled = {"lambda_cl": trial.suggest_float("lambda_cl", 1e-4, 0.1, log=True),
                           "mask_weight": trial.suggest_float("mask_weight", 0.0, 0.5),
                           "macr_alpha": trial.suggest_float("macr_alpha", 0.0, 1.0),
                           "macr_beta": trial.suggest_float("macr_beta", 0.0, 1.0)}
            else:
                sampled = {"cl_tau": trial.suggest_float("cl_tau", 0.05, 0.5),
                           "knn_k": trial.suggest_categorical("knn_k", [5, 10, 15, 20, 30]),
                           "n_layers": trial.suggest_int("n_layers", 0, 4)}
            reference = trial.suggest_categorical("cf_reference", args.cf_references) if args.tune_inference == "counterfactual" else 0.0
            values, times = [], []
            constructor_params = {k: v for k, v in sampled.items() if k != "cf_reference"}
            for seed in args.seeds:
                seed_everything(seed)
                if name == "DMCRec":
                    model = MyRec(**dmcrec_params(args, seed, args.variants[0], sampled=constructor_params))
                    model.inference_mode, model.cf_reference = args.tune_inference, reference
                else:
                    original = dict(args.overrides.get(name, {}))
                    args.overrides[name] = {**original, **constructor_params}
                    try:
                        model = build_baseline(name, args, seed, visual_mean)
                    finally:
                        args.overrides[name] = original
                started = time.perf_counter()
                model.fit(split.train_set, split.val_set)
                times.append(time.perf_counter() - started)
                validation = evaluate_recommendations(model, split.train_set, split.val_set, ks=(20,))
                if "NDCG@20" not in validation["metrics"]:
                    raise ValueError("Validation has no evaluable users")
                values.append(validation["metrics"]["NDCG@20"])
                del model
                gc.collect()
                if args.device == "cuda":
                    torch.cuda.empty_cache()
            trial.set_user_attr("validation_ndcg20_by_seed", values)
            trial.set_user_attr("train_seconds_by_seed", times)
            return float(np.mean(values))

        # --trials is the desired completed-study budget, including resumed trials.
        completed = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)
        study.optimize(objective, n_trials=max(0, args.trials - completed))
        report = {"study_name": study.study_name, "dataset": args.dataset, "setting": setting,
                  "model": name, "selection_split": "validation", "test_evaluated": False,
                  "best_validation_ndcg20": study.best_value, "best_params": study.best_params,
                  "completed_trials": sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials),
                  "inference_mode": args.tune_inference, "variant": args.variants[0],
                  "training_options": jsonable(vars(args)),
                  "data_report": split.preprocessing_report, "environment": environment}
        write_json(directory / "best.json", report)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        reports.append(report)
    return reports


if __name__ == "__main__":
    import json
    parser = make_parser()
    parser.description = __doc__
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--params-group", type=int, choices=[0, 1], default=0)
    parser.add_argument("--search-sampler", choices=["tpe", "grid"], default="tpe")
    parser.add_argument("--tune-inference", choices=["main", "factual", "counterfactual"], default="main")
    parser.add_argument("--search-param", action="append", default=[], metavar="KEY=JSON_LIST",
                        help='Categorical search, e.g. --search-param k=[64,128] for WMF')
    args = parse_args(parser=parser)
    if len(args.models) != 1 or len(args.variants) != 1:
        parser.error("Tune one model and one variant per study")
    if (args.val_size <= 0 and args.split_source == "random") or args.trials < 1:
        parser.error("Tuning requires --val-size > 0 and --trials > 0")
    args.search_space = {}
    for entry in args.search_param:
        try:
            key, raw = entry.split("=", 1)
            values = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            parser.error("Use --search-param KEY=JSON_LIST")
        if not isinstance(values, list) or not values or any(isinstance(v, (dict, list)) for v in values):
            parser.error("Search choices must be a nonempty list of scalar values")
        if key in {"name", "seed", "use_gpu", "device", "implementation", "inference_mode", "cf_reference"}:
            parser.error("Use dedicated options for " + key)
        args.search_space[key] = values
    if args.models[0] != "DMCRec" and not args.search_space:
        parser.error("Baseline tuning requires explicit --search-param ranges")
    tune(args)
