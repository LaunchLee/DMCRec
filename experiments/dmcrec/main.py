"""DMCRec experiment runner for local CPU, Colab and AutoDL.

Use --help / --dry-run to inspect a run without loading data or writing files.
Run from the repository root with python -m experiments.dmcrec.main, or run
this file directly. Historical papers and cache artifacts are never overwritten.
"""
import argparse
import hashlib
import importlib
import inspect
import json
import math
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
PROJECT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASELINES = {"MSCA": ("models.msca", "MSCA"), "MMGCF": ("models.mmgcf", "MMGCF"),
             "MENTOR": ("models.mentor", "MENTOR"), "DAMRS": ("models.damrs", "DAMRS"),
             "FREEDOM": ("models.freedom", "FREEDOM"), "BM3": ("models.bm3", "BM3"),
             "LayerGCN": ("models.layergcn", "LayerGCN"), "LATTICE": ("models.lattice", "LATTICE"),
             "MMGCN": ("models.mmgcn", "MMGCN"),
             **{name: ("cornac.models", name) for name in ["CausalRec", "VBPR", "BPR", "WMF", "MF"]}}
VARIANTS = {"full": {}, "woGCN": {"use_gcn": False}, "woKNN": {"use_knn_graph": False},
            "woDynKNN": {"epoch_refresh_knn": 0, "epoch_refresh_knn_auto": False,
                         "epoch_refresh_knn_auto_delay": False},
            "woCL": {"lambda_cl": 0.0}, "woMask": {"mask_weight": 0.0},
            "woText": {"use_text": False}, "woDebias": {"use_debias": False},
            "woMACR-A": {"macr_alpha": 0.0}, "woMACR-B": {"macr_beta": 0.0},
            "concat": {"fusion_mode": "concat"}}


def make_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--dataset", choices=["Baby", "Sports", "Clothing", "Elec", "Cornac-Clothing", "Bili", "Pixel200K", "MicroLens50K"], default="Baby")
    p.add_argument("--data-dir", type=Path, default=PROJECT / "datasets")
    p.add_argument("--cache-dir", type=Path, default=PROJECT / "cache")
    p.add_argument("--output-dir", type=Path, default=PROJECT / "cache" / "runs")
    p.add_argument("--models", nargs="+", choices=["DMCRec", *BASELINES], default=["DMCRec"])
    p.add_argument("--variants", nargs="+", choices=list(VARIANTS), default=["full"])
    p.add_argument("--seeds", nargs="+", type=int, default=[42])
    p.add_argument("--split-seed", type=int, default=42, help="Fixed independently of model seeds")
    p.add_argument("--split-source", choices=["random", "provided"], default="random",
                   help="MMRec only: provided explicitly uses x_label 0=train, 1=validation, 2=test")
    p.add_argument("--protocol", choices=["corrected", "legacy"], default="corrected")
    p.add_argument("--implementation", choices=["corrected", "legacy"], default="corrected")
    p.add_argument("--val-size", type=float, default=None, help="Defaults to 0.1; legacy protocol defaults to 0")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--setting", choices=["original", "sparse", "both"], default="original")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--feature-batch-size", type=int, default=256)
    p.add_argument("--embedding-dim", type=int, default=128)
    p.add_argument("--semantic-dim", type=int, default=64)
    p.add_argument("--learning-rate", type=float, default=0.001)
    p.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    p.add_argument("--ks", nargs="+", type=int, default=[20, 50])
    p.add_argument("--max-items", type=int, default=None, help="Subsample items for a smoke run, never use as full benchmark")
    p.add_argument("--k-core", type=int, default=None)
    p.add_argument("--knn-k", type=int, default=10)
    p.add_argument("--knn-chunk-size", type=int, default=1024)
    p.add_argument("--refresh", choices=["plateau", "static", "periodic", "warmup", "drift"], default="plateau")
    p.add_argument("--refresh-every", type=int, default=20)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--inference", nargs="+", choices=["main", "factual", "counterfactual"], default=["main", "factual", "counterfactual"])
    p.add_argument("--cf-references", nargs="+", type=float, default=[0.0], help="Matching-score reference candidates; select on validation NDCG@20")
    p.add_argument("--param", action="append", default=[], metavar="MODEL.KEY=JSON", help="Constructor override, e.g. DMCRec.lambda_cl=0.005 or WMF.k=64")
    p.add_argument("--save-model", action="store_true")
    p.add_argument("--resume", action="store_true", help="Skip completed matching run IDs")
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--dry-run", action="store_true", help="Print resolved options, without importing ML packages or accessing data")
    p.add_argument("--hf-endpoint", default=None, help="Optional Hugging Face endpoint; no mirror forced")
    p.add_argument("--selection-record", type=Path, default=None,
                   help="Validation-only best.json used to configure this evaluation")
    p.add_argument("--controlled-refresh", action="store_true",
                   help="Allow a declared refresh-strategy control against a selected plateau model")
    return p


def parse_args(argv=None, parser=None):
    p = parser or make_parser()
    args = p.parse_args(argv)
    if args.val_size is None:
        args.val_size = 0.0 if args.protocol == "legacy" else 0.1
    if args.split_source == "provided":
        if args.dataset not in {"Baby", "Sports", "Clothing", "Elec"} or args.protocol != "corrected":
            p.error("Provided x_label splits require an MMRec dataset and the corrected protocol")
    if not 0 < args.test_size < 1 or not 0 <= args.val_size < 1 or args.test_size + args.val_size >= 1:
        p.error("Invalid validation/test fractions")
    if args.epochs < 1 or args.batch_size < 1 or min(args.ks) < 1 or args.knn_k < 1 or args.refresh_every < 1:
        p.error("Epochs, batch size, cutoffs, KNN k and refresh interval must be positive")
    if len(args.cf_references) > 1 and args.val_size == 0 and args.split_source == "random":
        p.error("Selecting a counterfactual reference requires an independent validation set")
    if len(set(args.seeds)) != len(args.seeds):
        p.error("Seeds must be unique")
    for values in [args.models, args.variants, args.inference]:
        if len(set(values)) != len(values):
            p.error("Models, variants and inference modes must not contain duplicates")
    if min(args.embedding_dim, args.semantic_dim, args.knn_chunk_size, args.feature_batch_size) < 1 or args.warmup_epochs < 0:
        p.error("Dimensions and chunk/batch sizes must be positive; warmup must be nonnegative")
    if args.max_items is not None and args.max_items < 1:
        p.error("--max-items must be positive")
    if args.selection_record is not None:
        selection = json.loads(args.selection_record.read_text(encoding="utf-8"))
        if selection.get("selection_split") != "validation" or selection.get("test_evaluated") is not False:
            p.error("Selection record must come from validation-only tuning")
        if selection.get("dataset") != args.dataset or args.models != [selection.get("model")]:
            p.error("Selection record dataset/model does not match the requested evaluation")
    elif args.controlled_refresh:
        p.error("--controlled-refresh requires --selection-record")
    if not all(math.isfinite(c) for c in args.cf_references) or not math.isfinite(args.learning_rate) or args.learning_rate <= 0:
        p.error("References must be finite and learning rate must be positive")
    args.overrides = {}
    for value in args.param:
        try:
            address, raw = value.split("=", 1)
            model, key = address.split(".", 1)
            parsed = json.loads(raw)
        except (ValueError, json.JSONDecodeError):
            p.error("Use --param MODEL.KEY=JSON")
        if model not in ["DMCRec", *BASELINES]:
            p.error("Unknown model in --param: " + model)
        # Keep execution identity and controlled ablations outside ad-hoc overrides.
        if key in {"name", "seed", "use_gpu", "device", "implementation", "inference_mode", "cf_reference"}:
            p.error("Use dedicated options for " + key)
        args.overrides.setdefault(model, {})[key] = parsed
    return args


def dmcrec_params(args, seed, variant="full", sampled=None):
    params = dict(name="DMCRec_" + variant, k=args.embedding_dim, k2=args.semantic_dim,
                  n_epochs=args.epochs, batch_size=args.batch_size, learning_rate=args.learning_rate,
                  lambda_w=0.001, lambda_e=0.001, lambda_cl=0.01, cl_tau=0.2,
                  knn_k=args.knn_k, knn_chunk_size=args.knn_chunk_size, knn_sim_threshold=0.85,
                  epoch_refresh_knn=0, epoch_refresh_knn_auto=args.refresh in {"plateau", "warmup", "drift"},
                  epoch_refresh_knn_auto_delay=args.refresh == "plateau",
                  epoch_refresh_knn_auto_delay_flag=10,
                  warmup_epochs=args.warmup_epochs if args.refresh == "warmup" else 0,
                  n_layers=2, macr_alpha=0.5, macr_beta=0.5, use_scheduler=False,
                  implementation=args.implementation, use_gpu=args.device == "cuda",
                  verbose=not args.quiet, seed=seed)
    if args.refresh == "periodic":
        params["epoch_refresh_knn"] = args.refresh_every
    params.update(args.overrides.get("DMCRec", {}))
    params.update(sampled or {})
    params.update(VARIANTS[variant])
    return params


def load_data(args, setting):
    from experiments.dmcrec import exdata
    common = dict(cache_dir=str(args.cache_dir), seed=args.split_seed, protocol=args.protocol,
                  val_size=args.val_size, test_size=args.test_size, make_long_tail=setting == "sparse",
                  max_samples=args.max_items, k_core=args.k_core, batch_size=args.feature_batch_size,
                  verbose=not args.quiet)
    if args.dataset in {"Baby", "Sports", "Clothing", "Elec"}:
        return exdata.load_mmrec(datasets_dir=str(args.data_dir), which=args.dataset,
                                split_source=args.split_source, **common)
    if args.dataset == "Cornac-Clothing":
        return exdata.load_cornac_clothing(**common)
    if args.dataset == "Bili":
        return exdata.load_ninerec_bili(ninerec_dir=str(args.data_dir / "NineRec-Downstream-Bili"), **common)
    if args.dataset == "Pixel200K":
        return exdata.load_pixel200k(pixel200k_dir=str(args.data_dir / "Pixel200k"), **common)
    return exdata.load_microlens50k(mlens50k_dir=str(args.data_dir / "MicroLens-50k"), **common)


def build_baseline(name, args, seed, mm_visuals_mean):
    import numpy as np
    module, symbol = BASELINES[name]
    if module.startswith("models."):
        module = "experiments.dmcrec." + module
    cls = getattr(importlib.import_module(module), symbol)
    def construct(**kwargs):
        kwargs.update(args.overrides.get(name, {}))
        if "seed" in inspect.signature(cls).parameters:
            kwargs["seed"] = seed
        model = cls(**kwargs)
        model.experiment_params = kwargs
        return model
    # Original baseline presets; overrides are recorded in each manifest.
    NUM_FACTOR1, NUM_FACTOR2, NUM_FACTOR3 = args.embedding_dim, args.semantic_dim, args.semantic_dim // 2
    NUM_EPOCHS, BATCH_SIZE, VERBOSE, SEED = args.epochs, args.batch_size, not args.quiet, seed
    device = args.device
    if name == "MSCA":
        msca = construct(
            name="MSCA",
            embedding_size=NUM_FACTOR2,
            knn_k=10,
            n_layers=3,
            fusion_coeff=0.4,
            cl_weight=0.001,
            reg_weight=5e-8,
            tau=0.2,
            n_epochs=NUM_EPOCHS,
            learning_rate=args.learning_rate,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (msca)

    if name == "MMGCF":
        mmgcf = construct(
            embedding_size=NUM_FACTOR2,
            feat_embed_dim=NUM_FACTOR2,
            n_ui_layers=3,
            reg_weight=1e-4,
            fusion_mode="concat",
            weighting="alpha",
            dropout=0.3,
            learning_rate=args.learning_rate,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (mmgcf)

    if name == "MENTOR":
        mentor = construct(
            name="MENTOR",
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            embedding_size=NUM_FACTOR3,  # Inner it is fused to 2x size.
            # feat_embed_dim=NUM_FACTOR3,
            lr=args.learning_rate,
            dropout=0.5,
            reg_weight=0.001,
            mm_image_weight=0.5,
            n_mm_layers=1,
            device=device,
            verbose=VERBOSE
        )
        return (mentor)

    if name == "DAMRS":
        damrs = construct(
            embedding_size=NUM_FACTOR2,
            knn_k=10,
            n_mm_layers=1,
            n_ui_layers=2,
            kl_weight=1.0,
            neighbor_weight=0.001,
            learning_rate=args.learning_rate,
            batch_size=BATCH_SIZE,
            n_epochs=NUM_EPOCHS,
            verbose=VERBOSE,
            device=device
        )
        return (damrs)

    if name == "FREEDOM":
        freedom = construct(
            embedding_size=NUM_FACTOR2,
            feat_embed_dim=NUM_FACTOR2,
            knn_k=10,
            lambda_coeff=0.9,
            n_mm_layers=1,
            n_ui_layers=2,
            reg_weight=0.001,
            mm_image_weight=0.1,
            dropout=0.1,
            learning_rate=args.learning_rate,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (freedom)

    if name == "BM3":
        bm3 = construct(
            embedding_size=NUM_FACTOR2,
            n_layers=2,
            dropout=0.3,
            reg_weight=0.0001,
            cl_weight=0.3,
            learning_rate=args.learning_rate,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (bm3)

    if name == "LayerGCN":
        layer_gcn = construct(
            embedding_size=NUM_FACTOR2,
            n_layers=3,
            reg_weight=0.001,
            dropout=0.1,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (layer_gcn)

    if name == "LATTICE":
        lattice = construct(
            embedding_size=NUM_FACTOR2,
            feat_embed_dim=NUM_FACTOR2,
            knn_k=10,
            lambda_coeff=0.9,
            n_layers=1,
            reg_weight=0.001,
            learning_rate=args.learning_rate,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (lattice)

    if name == "CausalRec":
        causalrec = construct(
            k=NUM_FACTOR1,
            k2=NUM_FACTOR2,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            lambda_w=0.8,
            lambda_b=0.8,
            lambda_e=1.0,
            lambda_2=0.2,
            tanh=1,
            learning_rate=args.learning_rate,
            mean_feat=mm_visuals_mean / (np.linalg.norm(mm_visuals_mean) + 1e-9),
            use_gpu=args.device == "cuda",
            verbose=VERBOSE,
            seed=SEED
        )
        return (causalrec)

    if name == "MMGCN":
        mmgcn = construct(
            embedding_size=NUM_FACTOR2,
            reg_weight=0.0001,
            learning_rate=args.learning_rate,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=VERBOSE,
            device=device
        )
        return (mmgcn)

    if name == "VBPR":
        vbpr = construct(
            k=NUM_FACTOR1,
            k2=NUM_FACTOR2,
            n_epochs=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=args.learning_rate,
            use_gpu=args.device == "cuda",
            verbose=VERBOSE,
            seed=SEED
        )
        return (vbpr)

    if name == "BPR":
        bpr = construct(
            k=NUM_FACTOR1,
            max_iter=NUM_EPOCHS,
            learning_rate=args.learning_rate,
            lambda_reg=0.001,
            verbose=VERBOSE,
            seed=SEED
        )
        return (bpr)

    if name == "WMF":
        wmf = construct(
            k=NUM_FACTOR1,
            max_iter=NUM_EPOCHS,
            learning_rate=args.learning_rate,
            verbose=VERBOSE,
            seed=SEED
        )
        return (wmf)

    if name == "MF":
        mf = construct(
            k=NUM_FACTOR1,
            max_iter=NUM_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=args.learning_rate,
            lambda_reg=0.0,
            verbose=VERBOSE
        )
        return (mf)

    raise ValueError(name)


def jsonable(value):
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(v) for v in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    return value


def write_json(path, value):
    """Atomic completion marker; a failed run can be restarted with --resume."""
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(jsonable(value), ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def source_hashes():
    paths = ["main.py", "main_wo.py", "main_opt.py", "exdata.py", "exutils.py", "my_models/myrec.py"]
    paths += [str(p.relative_to(PROJECT)) for p in sorted((PROJECT / "models").glob("*.py"))]
    return {p: hashlib.sha256((PROJECT / p).read_bytes()).hexdigest() for p in paths}


def prepare_runtime(args):
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    import torch
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; select --device cpu for a small smoke run")
    args.device = "cuda" if args.device == "auto" and torch.cuda.is_available() else ("cpu" if args.device == "auto" else args.device)
    if args.device == "cpu":
        # Also honor CPU selection in the TensorFlow-based Cornac WMF backend.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    import importlib.metadata
    versions = {}
    for name in ["torch", "cornac", "numpy", "scipy", "pandas", "optuna", "torch-geometric", "tensorflow", "transformers", "sentence-transformers"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"python": sys.version, "packages": versions, "device": args.device,
            "gpu": torch.cuda.get_device_name(0) if args.device == "cuda" else None,
            "cuda": torch.version.cuda, "source_sha256": source_hashes()}


def select_reference(model, split, references):
    from experiments.dmcrec.exutils import evaluate_recommendations
    if len(references) == 1:
        model.cf_reference = references[0]
        return {"selected": references[0], "selection": "prespecified", "trials": []}
    if split.val_set is None:
        raise ValueError("Reference selection requires validation data")
    model.inference_mode = "counterfactual"
    trials = []
    for reference in references:
        model.cf_reference = reference
        score = evaluate_recommendations(model, split.train_set, split.val_set, ks=(20,))
        if "NDCG@20" not in score["metrics"]:
            raise ValueError("Validation has no evaluable users")
        trials.append({"reference": reference, "validation_ndcg20": score["metrics"]["NDCG@20"]})
    best = max(trials, key=lambda t: t["validation_ndcg20"])
    model.cf_reference = best["reference"]
    return {"selected": best["reference"], "selection": "validation_ndcg20", "trials": trials}


def summarize(records):
    import numpy as np
    groups = {}
    for record in records:
        for mode, result in record["evaluations"].items():
            key = (record["dataset"], record["setting"], record["model"], record["variant"], mode)
            groups.setdefault(key, []).append((record["seed"], result["metrics"]))
    summary = []
    for key, runs in groups.items():
        metrics = {}
        for metric in sorted({m for _, values in runs for m in values}):
            values = [values[metric] for _, values in runs if values.get(metric) is not None]
            metrics[metric] = {"mean": float(np.mean(values)),
                               "std": float(np.std(values, ddof=1)) if len(values) > 1 else None,
                               "n": len(values)} if values else None
        summary.append({"dataset": key[0], "setting": key[1], "model": key[2], "variant": key[3],
                        "inference": key[4], "seeds": [s for s, _ in runs], "metrics": metrics})
    # Independent training seeds are the units; do not pool users across seeds.
    comparisons = []
    from scipy.stats import ttest_rel
    for key, runs in groups.items():
        reference_key = (key[0], key[1], "DMCRec", "full", "main")
        if key == reference_key or reference_key not in groups:
            continue
        other, reference = dict(runs), dict(groups[reference_key])
        seeds = sorted(set(other) & set(reference))
        if len(seeds) < 2:
            continue
        for metric in ["Recall@20", "NDCG@20", "TailRecall@20", "ARP@20", "CatalogCoverage@20"]:
            if not all(metric in reference[s] and metric in other[s] for s in seeds):
                continue
            delta = np.array([reference[s][metric] - other[s][metric] for s in seeds])
            p_value = None
            if np.std(delta) > 0:
                p_value = float(ttest_rel([reference[s][metric] for s in seeds], [other[s][metric] for s in seeds]).pvalue)
            comparisons.append({"reference": list(reference_key), "comparison": list(key), "metric": metric,
                                "seeds": seeds, "mean_reference_minus_comparison": float(delta.mean()),
                                "paired_seed_ttest_p_uncorrected": p_value})
    return {"runs": summary, "comparisons": comparisons,
            "statistics_note": "Sample SD across training seeds on one fixed split; paired tests are exploratory, uncorrected for multiple comparisons. A single seed has no uncertainty estimate."}


def run(args):
    if args.dry_run:
        print(json.dumps(jsonable({"options": vars(args), "dmcrec": {v: dmcrec_params(args, args.seeds[0], v) for v in args.variants}}), ensure_ascii=False, indent=2))
        return []
    environment = prepare_runtime(args)
    import gc
    import torch
    from experiments.dmcrec.my_models.myrec import MyRec
    from experiments.dmcrec.exutils import seed_everything, evaluate_recommendations
    settings = ["original", "sparse"] if args.setting == "both" else [args.setting]
    records = []
    for setting in settings:
        split, visual_mean = load_data(args, setting)
        data_report = split.preprocessing_report
        selection_provenance = None
        if args.selection_record is not None:
            selected_bytes = args.selection_record.read_bytes()
            selected = json.loads(selected_bytes)
            if selected.get("setting") != setting or selected.get("data_report") != data_report:
                raise ValueError("Selected tuning data/protocol differs from this evaluation")
            if selected.get("environment", {}).get("source_sha256") != environment["source_sha256"]:
                raise ValueError("Selected tuning code differs from this evaluation")
            for key in ["epochs", "batch_size", "learning_rate", "embedding_dim", "semantic_dim",
                        "split_source", "protocol", "implementation", "refresh", "knn_k", "split_seed"]:
                if key == "refresh" and args.controlled_refresh:
                    continue
                if selected.get("training_options", {}).get(key) != getattr(args, key):
                    raise ValueError(f"Selected training option differs: {key}")
            selection_provenance = {"path": str(args.selection_record),
                                    "sha256": hashlib.sha256(selected_bytes).hexdigest(),
                                    "best_validation_ndcg20": selected["best_validation_ndcg20"]}
        print(json.dumps(jsonable({"dataset": args.dataset, "setting": setting, "data": data_report}), ensure_ascii=False))
        for name in args.models:
            for variant in (args.variants if name == "DMCRec" else ["full"]):
                for seed in args.seeds:
                    seed_everything(seed)
                    model = MyRec(**dmcrec_params(args, seed, variant)) if name == "DMCRec" else build_baseline(name, args, seed, visual_mean)
                    params = dmcrec_params(args, seed, variant) if name == "DMCRec" else model.experiment_params
                    if selection_provenance is not None:
                        for key, value in selected["best_params"].items():
                            if key == "cf_reference" or (name == "DMCRec" and key in VARIANTS[variant]):
                                continue
                            if params.get(key) != value:
                                raise ValueError(f"Evaluation did not apply selected parameter {key}={value}")
                    config = {k: v for k, v in vars(args).items()
                              if k not in {"resume", "dry_run", "output_dir", "quiet", "models", "variants", "seeds", "setting"}}
                    identity = jsonable({"options": config, "model": name, "variant": variant, "seed": seed,
                                        "setting": setting, "params": {k: v for k, v in params.items() if k != "verbose"},
                                        "data": data_report, "environment": environment,
                                        "selection": selection_provenance})
                    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
                    directory = args.output_dir / args.dataset / setting / name / variant / f"seed-{seed}-{digest}"
                    result_path = directory / "result.json"
                    if result_path.exists() and args.resume:
                        record = json.loads(result_path.read_text())
                        records.append(record)
                        print("Resumed completed run:", directory)
                        del model
                        continue
                    if directory.exists() and not args.resume:
                        raise FileExistsError(f"Run exists: {directory}. Use --resume or another --output-dir")
                    directory.mkdir(parents=True, exist_ok=True)
                    manifest = {"identity": identity, "environment": environment,
                                "selection": "fixed final epoch; no test-based checkpoint selection",
                                "negative_sampling": "uniform unobserved training items (DMCRec)",
                                "causal_scope": "model-based interventions; identification from observational data is not established"}
                    write_json(directory / "manifest.json", manifest)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                    started = time.perf_counter()
                    model.fit(split.train_set, split.val_set)
                    if args.device == "cuda":
                        torch.cuda.synchronize()
                    train_seconds = time.perf_counter() - started
                    backend = "tensorflow" if name == "WMF" else ("native_cpu" if name in {"BPR", "MF"} else "pytorch")
                    peak_memory = torch.cuda.max_memory_allocated() if args.device == "cuda" and backend == "pytorch" else None
                    reference = select_reference(model, split, args.cf_references) if name == "DMCRec" and model.use_debias and "counterfactual" in args.inference else None
                    modes = args.inference if name == "DMCRec" and model.use_debias else ["main"]
                    evaluations = {}
                    for mode in modes:
                        if name == "DMCRec":
                            model.inference_mode = mode
                        evaluations[mode] = evaluate_recommendations(model, split.train_set, split.test_set,
                                                                     ks=args.ks, exclude_set=split.val_set)
                    record = {"dataset": args.dataset, "setting": setting, "model": name, "variant": variant,
                              "seed": seed, "run_dir": str(directory), "train_seconds": train_seconds,
                              "peak_cuda_bytes": peak_memory, "training_backend": backend,
                              "memory_scope": "PyTorch allocator peak only; not process-wide GPU memory",
                              "reference_selection": reference,
                              "training_history": getattr(model, "training_history", []),
                              "refresh_history": getattr(model, "refresh_history", []),
                              "graph_drift_history": getattr(model, "graph_drift_history", []),
                              "graph_unlock_epoch": getattr(model, "graph_unlock_epoch", None),
                              "evaluations": evaluations}
                    if args.save_model:
                        if name == "DMCRec":
                            model.inference_mode = "main"
                        model.save(str(directory / "model"), metadata={"manifest": str(directory / "manifest.json")})
                    write_json(result_path, record)
                    records.append(record)
                    displayed = {m: {k: v for k, v in e["metrics"].items()
                                     if k in {f"Recall@{args.ks[0]}", f"NDCG@{args.ks[0]}",
                                              f"TailRecall@{args.ks[0]}", f"ARP@{args.ks[0]}"}}
                                 for m, e in evaluations.items()}
                    print(json.dumps({"run": str(directory), "metrics": displayed}, ensure_ascii=False))
                    del model
                    gc.collect()
                    if args.device == "cuda":
                        torch.cuda.empty_cache()
    summary = summarize(records)
    summary_key = hashlib.sha256(json.dumps(jsonable([r["run_dir"] for r in records]), sort_keys=True).encode()).hexdigest()[:12]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"summary-{summary_key}.json", summary)
    print("Summary:", args.output_dir / f"summary-{summary_key}.json")
    return records


if __name__ == "__main__":
    run(parse_args())
