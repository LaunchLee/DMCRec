"""Preflight, staged experiment commands and collection for a single GPU host."""
import argparse
import contextlib
import importlib.metadata
import json
import hashlib
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
import threading

ROOT = Path(__file__).resolve().parents[3]
PROJECT = ROOT / "experiments/dmcrec"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def read_plan(path):
    plan = json.loads(Path(path).read_text(encoding="utf-8"))
    if plan.get("schema") != 1:
        raise ValueError("Unsupported plan schema")
    return plan


def protocol_summary(plan):
    """Describe the deliberately small protocol without importing ML packages."""
    validation_fits = math.prod(len(v) for v in plan["search_spaces"]["DMCRec"].values())
    formal_fits = len(plan["variants"]) * len(plan["evaluation_seeds"])
    return {"dataset": plan["datasets"], "setting": plan["settings"],
            "batch_size_candidate": plan["batch_size"], "epochs": plan["epochs"],
            "validation_fits": validation_fits, "formal_fits": formal_fits,
            "total_model_fits": validation_fits + formal_fits,
            "grid": plan["search_spaces"]["DMCRec"],
            "fixed_parameters": plan.get("fixed_parameters", {}),
            "configuration_frozen": plan.get("execution_gate", {}).get("configuration_frozen", False),
            "benchmark_max_minutes": plan["focused_followup"]["benchmark_max_minutes"],
            "wall_clock_limit_hours": plan["focused_followup"]["total_wall_clock_budget_hours"],
            "price_cap_CNY": plan["focused_followup"]["estimated_price_cap_CNY"],
            "execution_requires": "fresh measured GPU protocol record; matching plan/code/environment; budget admission"}


def protocol_sources():
    from experiments.dmcrec.main import source_hashes
    sources = source_hashes()
    sources["tools/remote.py"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    return sources


def protocol_versions():
    return {"python": sys.version, **{name: importlib.metadata.version(name)
            for name in ["torch", "cornac", "numpy", "scipy", "pandas", "optuna", "torch-geometric"]}}


def gpu_identity():
    import csv
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=uuid,name,memory.total,driver_version",
                                      "--format=csv,noheader,nounits"], text=True)
    rows = list(csv.reader(output.splitlines(), skipinitialspace=True))
    if len(rows) != 1 or len(rows[0]) != 4:
        raise RuntimeError("The minimal protocol requires one explicitly identified GPU")
    return dict(zip(["uuid", "name", "memory_total_MiB", "driver"], [v.strip() for v in rows[0]]))


def execution_readiness(args, jobs):
    """Fail closed before data loading or GPU training; receipt must be measured."""
    plan = read_plan(args.plan)
    gate = plan.get("execution_gate", {})
    if args.stage == "pilot":
        return None
    if not gate.get("enabled"):
        raise RuntimeError("Formal execution requires the measured minimal-protocol gate; legacy queues are preview-only")
    if not gate.get("configuration_frozen") or plan["focused_followup"].get("batch_is_provisional", True):
        raise RuntimeError("Protocol is not frozen: measure the actual GPU and finalize batch/configuration first")
    if args.stage not in gate["allowed_stages"]:
        raise RuntimeError("Only the frozen tune and paired ablate stages may execute")
    if (plan["datasets"] != ["Baby"] or plan["settings"] != ["original"] or plan["models"] != ["DMCRec"]
            or plan["tuning_seeds"] != [42] or plan["evaluation_seeds"] != [42, 43, 44, 45, 46]
            or plan["variants"] != ["full", "woDebias"] or plan["epochs"] != 200
            or plan.get("search_sampler") != "grid" or plan["trials"] != 2
            or plan["search_spaces"] != {"DMCRec": {"lambda_cl": [0.005, 0.01]}}):
        raise RuntimeError("Plan differs from the prespecified minimal validation/paired protocol")
    if args.device != "cuda" or args.use_presets:
        raise RuntimeError("Formal execution requires CUDA and new validation selection, not exploratory presets")
    if args.protocol_record is None or not args.protocol_record.is_file():
        raise RuntimeError("A real GPU measurement --protocol-record is required; do not invent timing evidence")
    raw = args.protocol_record.read_bytes()
    record = json.loads(raw)
    if record.get("schema") != 1 or record.get("plan_sha256") != hashlib.sha256(args.plan.read_bytes()).hexdigest():
        raise RuntimeError("Protocol record does not match the frozen plan")
    if record.get("source_sha256") != protocol_sources() or record.get("package_versions") != protocol_versions():
        raise RuntimeError("Protocol measurement code/environment differs from this run")
    if record.get("gpu") != gpu_identity():
        raise RuntimeError("GPU or driver changed; repeat the short measurement")
    now = time.time()
    def positive(value, name):
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise RuntimeError(f"Invalid measured {name}")
        return float(value)
    measured_at = positive(record.get("measured_at_unix"), "timestamp")
    started_at = positive(record.get("benchmark_started_at_unix"), "benchmark start")
    if not started_at <= measured_at <= now + 60 or now - measured_at > gate["max_record_age_seconds"]:
        raise RuntimeError("GPU measurement is stale or has invalid timestamps")
    if measured_at - started_at > plan["focused_followup"]["benchmark_max_minutes"] * 60:
        raise RuntimeError("Short measurement exceeded its 15-minute budget")
    measured = record.get("benchmark", {})
    if (measured.get("batch_size") != plan["batch_size"] or "max_items" not in measured or measured.get("max_items") is not None
            or measured.get("full_training_catalog") is not True or measured.get("test_evaluated") is not False
            or not measured.get("data_fingerprint")):
        raise RuntimeError("Measurement must use the actual batch/full training catalog and no test evaluation")
    expected_parameters = {**plan.get("fixed_parameters", {}),
                           "lambda_cl": max(plan["search_spaces"]["DMCRec"]["lambda_cl"])}
    if measured.get("parameters") != expected_parameters:
        raise RuntimeError("Benchmark parameters differ from the frozen full-model configuration")
    epochs = measured.get("epoch_seconds", [])
    if len(epochs) < plan["focused_followup"]["benchmark_protocol"]["minimum_epochs"]:
        raise RuntimeError("At least two measured full-data epochs are required")
    epoch_upper = max(positive(v, "epoch time") for v in epochs)
    overhead = max(positive(measured.get("run_overhead_seconds_upper"), "run overhead"),
                   gate["minimum_fixed_run_overhead_seconds"])
    run_upper = epoch_upper * plan["epochs"] * gate["epoch_time_safety_multiplier"] + overhead
    price = positive(record.get("hourly_price_CNY"), "current hourly price")
    followup = plan["focused_followup"]
    allowed_seconds = min(followup["total_wall_clock_budget_hours"] * 3600,
                          followup["estimated_price_cap_CNY"] / price * 3600)
    deadline = started_at + allowed_seconds
    summary = protocol_summary(plan)
    fits_left = summary["total_model_fits"] if args.stage == "tune" else len(jobs)
    if now + fits_left * run_upper > deadline:
        raise RuntimeError("Projected remaining protocol exceeds the time/price cap; do not launch a long queue")
    for dataset, setting, model, variant, _ in jobs:
        if dataset != "Baby" or setting != "original" or model != "DMCRec" or variant not in {"tune", "full", "woDebias"}:
            raise RuntimeError("Requested job exceeds the minimal Baby/full/woDebias protocol")
    if args.stage == "ablate":
        selected_path, selected = find_selection(args.selection_dir, "Baby", "original", "DMCRec")
        options = selected.get("training_options", {})
        for key in ["batch_size", "epochs", "learning_rate", "split_seed", "split_source", "refresh",
                    "embedding_dim", "semantic_dim", "knn_k", "knn_chunk_size", "val_size", "test_size",
                    "refresh_every", "warmup_epochs"]:
            if options.get(key) != plan[key]:
                raise RuntimeError(f"Validation selection differs from frozen {key}")
        if selected.get("environment", {}).get("source_sha256") != {k: v for k, v in protocol_sources().items() if k != "tools/remote.py"}:
            raise RuntimeError("Validation selection uses stale model/data code")
        if options.get("search_sampler") != "grid" or selected.get("completed_trials") != summary["validation_fits"]:
            raise RuntimeError("Need the new complete validation grid; discarded selections are not reusable")
        if (selected.get("inference_mode") != "main"
                or options.get("overrides", {}).get("DMCRec") != plan.get("fixed_parameters", {})
                or options.get("search_space") != plan["search_spaces"]["DMCRec"]
                or options.get("trials") != plan["trials"]
                or set(selected["best_params"]) != {"lambda_cl"}
                or selected["best_params"]["lambda_cl"] not in plan["search_spaces"]["DMCRec"]["lambda_cl"]):
            raise RuntimeError("Validation objective/parameters differ from the prespecified full-model selection")
        if selected.get("data_report", {}).get("fingerprint") != measured["data_fingerprint"]:
            raise RuntimeError("Validation data differs from the full-data benchmark")
    return {"schema": 1, "protocol_record_sha256": hashlib.sha256(raw).hexdigest(),
            "deadline_unix": deadline, "checked_at_unix": now, "stage": args.stage,
            "projected_remaining_seconds": fits_left * run_upper, "hourly_price_CNY": price,
            "scope": "tool deadline only; stop instance separately to stop billing"}


def preflight(args):
    from experiments.dmcrec.tools.package_remote import inventory, sha256_file
    report = {"python": sys.version, "device_requested": args.device, "files_checked": 0,
              "hashes_checked": 0, "packages": {}}
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError("Use Python 3.11 for the tested remote environment")
    manifest = ROOT / "dmcrec_data_manifest.json"
    entries = json.loads(manifest.read_text())["files"] if manifest.exists() else [
        {"path": p.relative_to(ROOT).as_posix(), "bytes": p.stat().st_size} for p in inventory("data")]
    for entry in entries:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise RuntimeError(f"Missing/changed bundled data: {entry['path']}")
        report["files_checked"] += 1
        if args.verify_hashes:
            if "sha256" not in entry:
                raise RuntimeError("Full hash verification requires the extracted data manifest")
            if sha256_file(path) != entry["sha256"]:
                raise RuntimeError(f"Data hash mismatch: {entry['path']}")
            report["hashes_checked"] += 1
    for name in ["torch", "cornac", "numpy", "scipy", "pandas", "joblib", "optuna", "torch-geometric", "tensorflow"]:
        report["packages"][name] = importlib.metadata.version(name)
    import torch
    import numpy as np
    report["cuda_available"] = torch.cuda.is_available()
    if args.device == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; do not launch formal runs")
        device = torch.device("cuda")
        # Verify the sparse operation used by the backbone, not only nvidia-smi.
        a = torch.sparse_coo_tensor(torch.tensor([[0, 1], [1, 0]], device=device),
                                    torch.ones(2, device=device), (2, 2)).coalesce()
        result = torch.sparse.mm(a, torch.eye(2, device=device))
        torch.cuda.synchronize()
        if not torch.isfinite(result).all():
            raise RuntimeError("CUDA sparse multiplication produced invalid values")
        report["gpu"] = torch.cuda.get_device_name(0)
        report["gpu_total_bytes"] = torch.cuda.get_device_properties(0).total_memory
        report["torch_cuda"] = torch.version.cuda
    for path in sorted((PROJECT / "datasets").glob("MMRec-*/image_feat.npy")):
        array = np.load(path, mmap_mode="r", allow_pickle=False)
        if array.dtype != np.float32 or array.ndim != 2 or array.shape[1] != 4096:
            raise RuntimeError(f"Unexpected Amazon image feature format: {path}")
    for dataset in ["Bili", "Pixel200K", "MicroLens50K"]:
        require_verified_text(dataset, args)
    report["notes"] = ["Bundled text matrices have SHA-bound row-ID sidecars recovered from the original encoder input order; no re-encoding is required.",
                       "Row indexing follows each dataset's explicit ID sequence, not arbitrary numeric raw IDs.",
                       "CUDA training is verified separately by the smoke command; this is not a formal experiment result."]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    path = args.output_dir / "preflight.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def common_flags(plan, args, dataset, setting):
    values = {"dataset": dataset, "data-dir": str(PROJECT / "datasets"),
              "cache-dir": str(args.cache_dir), "device": args.device,
              "protocol": "corrected", "implementation": "corrected", "setting": setting}
    for key in ["split_seed", "split_source", "val_size", "test_size", "epochs", "batch_size",
                "embedding_dim", "semantic_dim", "learning_rate", "knn_k", "knn_chunk_size",
                "refresh", "refresh_every", "warmup_epochs"]:
        values[key.replace("_", "-")] = plan[key]
    flags = []
    for key, value in values.items():
        flags += ["--" + key, str(value)]
    flags += ["--ks", *map(str, plan["ks"]), "--resume"]
    return flags


def find_selection(selection_dir, dataset, setting, model):
    matches = []
    for path in Path(selection_dir).rglob("best.json"):
        record = json.loads(path.read_text())
        if (record.get("dataset"), record.get("setting"), record.get("model"), record.get("variant")) == (dataset, setting, model, "full"):
            if record.get("selection_split") != "validation" or record.get("test_evaluated") is not False:
                raise RuntimeError(f"Non-validation selection: {path}")
            matches.append((path, record))
    if len(matches) != 1:
        raise RuntimeError(f"Need one curated validation best.json for {dataset}/{setting}/{model}, found {len(matches)}")
    return matches[0]


def planned_jobs(args):
    plan = read_plan(args.plan)
    datasets = args.datasets or plan["datasets"]
    models = args.models or plan["models"]
    for dataset in datasets:
        if dataset not in plan["datasets"] + plan["supplementary_datasets"]:
            raise ValueError(f"Unknown dataset: {dataset}")
    jobs = []
    settings = args.settings or plan["settings"]
    for dataset in datasets:
        for setting in settings:
            for model in (["DMCRec"] if args.stage in {"ablate", "refresh"} else models):
                flags = common_flags(plan, args, dataset, setting)
                if model == "DMCRec":
                    for key, value in plan.get("fixed_parameters", {}).items():
                        flags += ["--param", f"DMCRec.{key}=" + json.dumps(value)]
                if args.stage == "pilot":
                    flags[flags.index("--epochs") + 1] = "2"
                    flags += ["--max-items", "256", "--seeds", "42", "--inference", "main"]
                    jobs.append((dataset, setting, model, "pilot", [sys.executable, "-B", "-m", "experiments.dmcrec.main", *flags,
                                 "--models", model, "--output-dir", str(args.output_dir / "pilot")]))
                    continue
                if args.stage == "tune":
                    if model not in plan["search_spaces"]:
                        raise ValueError(f"No declared search space: {model}")
                    flags += ["--models", model, "--seeds", *map(str, plan["tuning_seeds"]),
                              "--trials", str(plan["trials"]), "--search-sampler", plan.get("search_sampler", "tpe"),
                              "--output-dir", str(args.output_dir / "selection")]
                    for key, values in plan["search_spaces"][model].items():
                        flags += ["--search-param", key + "=" + json.dumps(values, separators=(",", ":"))]
                    jobs.append((dataset, setting, model, "tune", [sys.executable, "-B", "-m", "experiments.dmcrec.main_opt", *flags]))
                    continue
                if args.selection_dir is None and not args.use_presets:
                    raise ValueError("Final/control stages require --selection-dir, or explicit --use-presets for exploratory runs")
                if args.selection_dir is not None:
                    path, record = find_selection(args.selection_dir, dataset, setting, model)
                    flags += ["--selection-record", str(path)]
                    for key, value in record["training_options"].get("overrides", {}).get(model, {}).items():
                        flags += ["--param", f"{model}.{key}=" + json.dumps(value)]
                    for key, value in record["best_params"].items():
                        if key == "cf_reference":
                            continue
                        flags += ["--param", f"{model}.{key}=" + json.dumps(value)]
                variants = plan["variants"] if args.stage == "ablate" else (["full"] if args.stage != "refresh" else plan["refresh_controls"])
                for variant in variants:
                    variant_flags = list(flags)
                    if args.stage == "refresh":
                        variant_flags[variant_flags.index("--refresh") + 1] = variant
                        if args.selection_dir:
                            variant_flags += ["--controlled-refresh"]
                    else:
                        variant_flags += ["--variants", variant]
                    for seed in plan["evaluation_seeds"]:
                        final_flags = [*variant_flags, "--models", model, "--seeds", str(seed),
                                       "--cf-references", *map(str, plan["cf_references"]),
                                       "--output-dir", str(args.output_dir / args.stage)]
                        if args.save_models and seed == plan["evaluation_seeds"][0] and variant == "full" and model in {"DMCRec", "MENTOR", "MSCA"}:
                            final_flags += ["--save-model"]
                        jobs.append((dataset, setting, model, variant, [sys.executable, "-B", "-m", "experiments.dmcrec.main", *final_flags]))
    if args.stage == "ablate" and plan.get("paired_seed_order"):
        jobs.sort(key=lambda job: (datasets.index(job[0]), settings.index(job[1]),
                  plan["evaluation_seeds"].index(int(job[-1][job[-1].index("--seeds") + 1])),
                  plan["variants"].index(job[3])))
    return jobs


def prepare_text(args):
    """Re-encode text on the remote GPU, preserving all historical caches."""
    import shutil
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    from experiments.dmcrec.main import load_data, parse_args
    if args.cache_dir.resolve() == (PROJECT / "cache").resolve():
        raise ValueError("Use a separate --cache-dir; historical caches must remain intact")
    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested for text encoding but unavailable")
    os.environ["HF_HUB_OFFLINE"] = "0"
    os.environ["TRANSFORMERS_OFFLINE"] = "0"
    if args.hf_endpoint:
        os.environ["HF_ENDPOINT"] = args.hf_endpoint
    destination = args.cache_dir / "_embeddings"
    destination.mkdir(parents=True, exist_ok=True)
    for path in (PROJECT / "cache/_embeddings").glob("*visual_embeddings.joblib"):
        target = destination / path.name
        if not target.exists():
            shutil.copy2(path, target)
    for dataset in args.datasets or ["Bili"]:
        if dataset not in {"Bili", "Pixel200K", "MicroLens50K"}:
            raise ValueError("Amazon MMRec already supplies encoded text; select a video dataset")
        options = parse_args(["--dataset", dataset, "--cache-dir", str(args.cache_dir),
                              "--device", args.device, "--max-items", "256", "--split-seed", "42"])
        print(f"Encoding verified text cache: {dataset}; training is not started", flush=True)
        load_data(options, "original")


def require_verified_text(dataset, args):
    names = {"Bili": "ninerec_downstream_bili_text_embeddings.joblib",
             "Pixel200K": "pixel200k_text_embeddings.joblib",
             "MicroLens50K": "microlens50k_text_embeddings.joblib"}
    if dataset not in names or args.allow_legacy_text_cache:
        return
    import joblib
    path = args.cache_dir / "_embeddings" / names[dataset]
    if not path.is_file():
        raise RuntimeError(f"Missing text cache for {dataset}; restore the bundled cache or explicitly encode a replacement: {path}")
    value = joblib.load(path)
    if isinstance(value, dict) and all(k in value for k in ["ids", "text_sha256", "model", "embeddings"]):
        return
    from experiments.dmcrec.tools.package_remote import sha256_file
    mapping = path.with_suffix(".ids.json")
    if mapping.is_file():
        metadata = json.loads(mapping.read_text())
        if metadata.get("cache_sha256") == sha256_file(path) and len(metadata.get("ids", [])) == len(value):
            return
    raise RuntimeError(f"Text cache has no valid row-ID mapping: {path}; provide its original encoding-order mapping or explicitly re-encode it")


@contextlib.contextmanager
def campaign_lock(output_dir):
    import fcntl
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / ".dmcrec_campaign.lock").open("a+") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("Another DMCRec campaign is already using this output directory") from exc
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def run_logged_job(command, log, environment, deadline=None):
    """Drain output while a separate timer enforces the shared protocol deadline."""
    expired = threading.Event()
    with log.open("w", encoding="utf-8") as handle:
        handle.write(shlex.join(command) + "\n")
        handle.flush()
        if deadline is not None and time.time() >= deadline:
            raise TimeoutError("Protocol deadline reached before job launch")
        child = subprocess.Popen(command, cwd=ROOT, env=environment, stdout=subprocess.PIPE,
                                 stderr=subprocess.STDOUT, text=True, start_new_session=True)
        def terminate_group():
            if child.poll() is None:
                expired.set()
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(child.pid, signal.SIGTERM)
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    with contextlib.suppress(ProcessLookupError):
                        os.killpg(child.pid, signal.SIGKILL)
        timer = threading.Timer(max(0, deadline - time.time()), terminate_group) if deadline is not None else None
        if timer:
            timer.daemon = True
            timer.start()
        try:
            for line in child.stdout:
                handle.write(line)
                handle.flush()
                print(line, end="", flush=True)
            status = child.wait()
        except BaseException:
            terminate_group()
            child.wait()
            raise
        finally:
            if timer:
                timer.cancel()
            child.stdout.close()
        if expired.is_set():
            raise TimeoutError(f"Protocol deadline reached; unfinished job terminated; inspect {log}")
        if status:
            raise RuntimeError(f"Job failed with exit {status}; inspect {log}")


def execute(args, jobs):
    readiness = execution_readiness(args, jobs)
    if readiness is not None:
        from experiments.dmcrec.main import write_json
        args.output_dir.mkdir(parents=True, exist_ok=True)
        write_json(args.output_dir / f"budget-admission-{args.stage}.json", readiness)
    deadline = readiness["deadline_unix"] if readiness is not None else None
    logs = args.output_dir / "logs" / args.stage
    logs.mkdir(parents=True, exist_ok=True)
    environment = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
                   "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    # Each job gets a fresh process; TensorFlow WMF cannot retain a GPU context
    # in the next PyTorch training job. Model failures stop the queue.
    verified = set()
    for number, (dataset, setting, model, variant, command) in enumerate(jobs, 1):
        if args.stage != "pilot" and dataset not in verified:
            require_verified_text(dataset, args)
            verified.add(dataset)
        stamp = time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + f"-{time.time_ns() % 1000000000:09d}"
        log = logs / f"{stamp}-{dataset}-{setting}-{model}-{variant}.log"
        print(f"[{number}/{len(jobs)}] {shlex.join(command)}\nLog: {log}", flush=True)
        run_logged_job(command, log, environment, deadline)


def collect(args):
    from experiments.dmcrec.main import summarize, write_json
    records, seen, protocols = [], set(), {}
    for path in sorted((args.output_dir / args.stage).rglob("result.json")):
        record = json.loads(path.read_text())
        key = tuple(record[k] for k in ["dataset", "setting", "model", "variant", "seed"])
        # Refresh controls share variant=full; distinguish their requested strategy.
        manifest = json.loads((path.parent / "manifest.json").read_text())
        identity = manifest["identity"]
        if args.stage == "refresh":
            record["variant"] = identity["options"]["refresh"]
            key = (*key, record["variant"])
        if key in seen:
            raise RuntimeError(f"Multiple configurations for the same evaluation cell: {key}; curate the output directory")
        seen.add(key)
        protocol_key = (record["dataset"], record["setting"])
        signature = json.dumps({"data": identity["data"], "source": identity["environment"]["source_sha256"]}, sort_keys=True)
        if protocol_key in protocols and protocols[protocol_key] != signature:
            raise RuntimeError(f"Cannot combine different data/code protocols: {protocol_key}")
        protocols[protocol_key] = signature
        for result in record["evaluations"].values():
            result.pop("per_user", None)  # Keep raw per-user results in the original files.
        records.append(record)
    if not records:
        raise ValueError("No completed records to collect")
    result = summarize(records)
    result["timing"] = [{k: record.get(k) for k in ["dataset", "setting", "model", "variant", "seed", "train_seconds", "peak_cuda_bytes", "training_backend"]} for record in records]
    destination = args.output_dir / f"{args.stage}_summary.json"
    write_json(destination, result)
    print(f"Collected {len(records)} completed runs: {destination}")
    return result


def baseline_spec(path):
    if path is None:
        raise RuntimeError('--followup-config is required for the separate baseline stage')
    value = json.loads(Path(path).read_text())
    spec = value.get('next_experiment_plan', value)
    baseline_design(spec)
    return spec


def baseline_design(spec):
    """Only the proposed Baby stage is supported; Sports remains separately gated."""
    if spec.get('id') != 'followup_v3_design':
        raise RuntimeError('Unknown follow-up design')
    common = spec.get('common_options', {})
    if (common.get('epochs') != 200 or common.get('split_seed') != 42
            or common.get('split_source') != 'random' or common.get('val_size') != .1
            or common.get('test_size') != .2):
        raise RuntimeError('Baseline comparison requires the fixed 70/10/20, seed42, 200-epoch protocol')
    jobs = [j for j in spec.get('jobs', []) if j.get('stage') == 'baby_baselines']
    if len(jobs) != 2 or [j.get('model') for j in jobs] != ['WMF', 'MENTOR']:
        raise RuntimeError('Baseline stage must contain exactly WMF and MENTOR')
    for job in jobs:
        model = job['model']
        grid = {'learning_rate' if model == 'WMF' else 'lr': [.0001, .001]}
        allowed_batches = [128, 64] if model == 'WMF' else [8192, 4096, 2048]
        if (job.get('dataset') != 'Baby' or job.get('search_grid') != grid
                or job.get('selection_fits') != 2 or job.get('formal_fits') != 5
                or job.get('evaluation_seeds') != [42, 43, 44, 45, 46]
                or job.get('evaluation_variants') != ['full']
                or job.get('batch_size_candidate') not in allowed_batches
                or job.get('fixed_parameters', {}).get('batch_size') != job.get('batch_size_candidate')
                or not job.get('data_fingerprint')):
            raise RuntimeError('Baseline job differs from the limited prespecified scope/batch')
    fields = ['dataset', 'model', 'batch_size_candidate', 'fixed_parameters', 'search_grid',
              'data_fingerprint', 'selection_fits', 'formal_fits', 'evaluation_seeds']
    return {'common_options': common, 'jobs': [{k: j[k] for k in fields} for j in jobs]}


def baseline_digest(spec):
    return hashlib.sha256(json.dumps(baseline_design(spec), sort_keys=True).encode()).hexdigest()


def baseline_versions():
    return {**protocol_versions(), 'tensorflow': importlib.metadata.version('tensorflow')}


def baseline_flags(job, spec, args, epochs=200):
    flags = ['--dataset', 'Baby', '--setting', 'original', '--models', job['model'],
             '--variants', 'full', '--protocol', 'corrected', '--implementation', 'corrected',
             '--device', 'cuda', '--data-dir', str(PROJECT / 'datasets'),
             '--cache-dir', str(args.cache_dir), '--batch-size', str(job['batch_size_candidate']),
             '--inference', 'main', '--ks', '20', '50']
    for key, value in spec['common_options'].items():
        flags += ['--' + key.replace('_', '-'), str(epochs if key == 'epochs' else value)]
    for key, value in job['fixed_parameters'].items():
        flags += ['--param', job['model'] + '.' + key + '=' + json.dumps(value)]
    return flags


def _positive(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
        raise RuntimeError('Invalid positive measurement/budget: ' + label)
    return float(value)


def baseline_projection(spec, record, stage, now=None):
    """Validate genuine per-candidate receipts before estimating any formal queue."""
    baseline_design(spec)
    if stage not in {'tune', 'evaluate'}:
        raise RuntimeError('Baseline stage only supports tune/evaluate')
    now = time.time() if now is None else now
    if (record.get('schema') != 1 or record.get('design_sha256') != baseline_digest(spec)
            or record.get('source_sha256') != protocol_sources()
            or record.get('package_versions') != baseline_versions() or record.get('gpu') != gpu_identity()):
        raise RuntimeError('Baseline measurement design/code/packages/GPU differ')
    start = _positive(record.get('benchmark_started_at_unix'), 'benchmark start')
    end = _positive(record.get('measured_at_unix'), 'measurement end')
    if not start <= end <= now + 60 or end - start > 900 or now - end > 86400:
        raise RuntimeError('Baseline measurement is stale, future-dated or exceeded 900 seconds')
    price = _positive(record.get('hourly_price_CNY'), 'hourly price')
    budget = spec['stages']['baby_baselines']
    hours = _positive(budget.get('proposed_wall_clock_cap_hours'), 'hours')
    cap = _positive(budget.get('proposed_price_cap_CNY'), 'price cap')
    if hours > 4.5 or cap > 7.11:
        raise RuntimeError('Baseline budget exceeds the user-approved 4.5h/7.11 CNY stage cap')
    deadline = start + min(hours * 3600, cap / price * 3600)
    profiles = record.get('profiles', [])
    if len(profiles) != 4:
        raise RuntimeError('Need all four model/learning-rate profiles, not DMCRec or sampled-item timings')
    bounds, seen = {}, set()
    for profile in profiles:
        model, candidate = profile.get('model'), profile.get('candidate')
        job = next((j for j in baseline_design(spec)['jobs'] if j['model'] == model), None)
        if job is None or candidate not in [.0001, .001] or (model, candidate) in seen:
            raise RuntimeError('Unexpected or duplicate measurement profile')
        seen.add((model, candidate))
        key = next(iter(job['search_grid']))
        params = profile.get('parameters', {})
        if (profile.get('batch_size') != job['batch_size_candidate']
                or profile.get('max_items', 'missing') is not None
                or profile.get('full_training_catalog') is not True or profile.get('test_evaluated') is not False
                or profile.get('data_fingerprint') != job['data_fingerprint']
                or profile.get('source_sha256') != record['source_sha256']
                or profile.get('package_versions') != record['package_versions']
                or profile.get('gpu') != record['gpu'] or params.get(key) != candidate
                or any(params.get(k) != v for k, v in job['fixed_parameters'].items())
                or params.get('max_iter' if model == 'WMF' else 'n_epochs') != 3):
            raise RuntimeError('Measured data, constructor or runtime does not match baseline design')
        epochs = profile.get('epoch_seconds', [])
        if len(epochs) != 3:
            raise RuntimeError('Each candidate requires three actually timed complete epochs')
        peak = _positive(profile.get('peak_device_memory_MiB'), 'device memory')
        if profile.get('memory_scope') != 'device-wide absolute nvidia-smi, not framework allocator or isolated process':
            raise RuntimeError('Memory measurement scope is missing')
        bound = max(_positive(v, 'epoch seconds') for v in epochs) * 200 * 1.5
        bound += max(120, _positive(profile.get('run_overhead_seconds_upper'), 'overhead'))
        bounds[model] = max(bounds.get(model, 0), bound)
    fits = 7 if stage == 'tune' else 5
    projected = sum(bounds.values()) * fits
    return {'schema': 1, 'stage': stage, 'design_sha256': baseline_digest(spec),
            'deadline_unix': deadline, 'checked_at_unix': now, 'hourly_price_CNY': price,
            'run_upper_seconds_by_model': bounds, 'remaining_fits': fits * 2,
            'projected_remaining_seconds': projected, 'within_budget': now + projected <= deadline,
            'formal_execution_authorized': spec.get('execution_enabled') is True
                and spec.get('configuration_frozen') is True and budget.get('budget_approved') is True,
            'scope': 'shared stage deadline from first measurement; does not stop AutoDL billing'}


class TimedEpochIterator:
    def __init__(self, iterable, seconds, synchronize=lambda: None):
        self.iterable, self.seconds, self.synchronize = iterable, seconds, synchronize

    def __getattr__(self, name):
        return getattr(self.iterable, name)

    def __iter__(self):
        for epoch in self.iterable:
            self.synchronize()
            tick = time.perf_counter()
            yield epoch
            self.synchronize()
            self.seconds.append(time.perf_counter() - tick)


@contextlib.contextmanager
def baseline_epoch_meter(model, seconds, synchronize):
    from unittest.mock import patch
    if model == 'WMF':
        from cornac.models.wmf import recom_wmf
        original = recom_wmf.trange
        with patch.object(recom_wmf, 'trange', lambda *a, **kw: TimedEpochIterator(original(*a, **kw), seconds)):
            yield
    else:
        import tqdm.auto
        original = tqdm.auto.tqdm
        def factory(*a, **kw):
            bar = original(*a, **kw)
            synchronize()
            tick, closed, old_close = time.perf_counter(), False, bar.close
            def close():
                nonlocal closed
                old_close()
                if not closed:
                    synchronize()
                    seconds.append(time.perf_counter() - tick)
                    closed = True
            bar.close = close
            return bar
        with patch.object(tqdm.auto, 'tqdm', factory):
            yield


def baseline_measure(args):
    """One isolated, three-epoch profile. Never evaluates test rankings."""
    started = _positive(args.benchmark_started, 'shared benchmark start')
    delay = started + 900 - time.time()
    if delay <= 0 or delay > 900:
        raise RuntimeError('Worker benchmark deadline is invalid')
    timer = threading.Timer(delay, lambda: os.kill(os.getpid(), signal.SIGTERM))
    timer.daemon = True
    timer.start()
    tick = time.perf_counter()
    from experiments.dmcrec.main import parse_args, prepare_runtime, load_data, build_baseline, jsonable, write_json
    from experiments.dmcrec.exutils import seed_everything, evaluate_recommendations
    spec = baseline_spec(args.followup_config)
    job = next((j for j in baseline_design(spec)['jobs'] if args.models == [j['model']]), None)
    if job is None or args.candidate_value not in [.0001, .001]:
        raise RuntimeError('One known model and learning-rate candidate required')
    key = next(iter(job['search_grid']))
    options = parse_args([*baseline_flags(job, spec, args, epochs=3), '--seeds', '42', '--quiet',
                          '--param', job['model'] + '.' + key + '=' + str(args.candidate_value)])
    prepare_runtime(options)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    split, visual = load_data(options, 'original')
    if split.preprocessing_report['fingerprint'] != job['data_fingerprint']:
        raise RuntimeError('Loaded baseline data fingerprint differs')
    seed_everything(42)
    model = build_baseline(job['model'], options, 42, visual)
    # Match main.run: finish PyTorch's lazy deterministic initialization before
    # importing the TensorFlow backend, then leave its runtime isolated here.
    if job['model'] == 'WMF':
        import tensorflow as tf
        if not tf.config.list_physical_devices('GPU'):
            raise RuntimeError('TensorFlow GPU unavailable; no CPU fallback measurement')
    stop, samples = threading.Event(), []
    def sample():
        while not stop.is_set():
            try:
                raw = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used',
                      '--format=csv,noheader,nounits'], text=True, timeout=3)
                samples.append(float(raw.strip()))
            except (ValueError, subprocess.SubprocessError, OSError):
                pass
            stop.wait(.5)
    thread = threading.Thread(target=sample, daemon=True)
    thread.start()
    seconds = []
    sync = torch.cuda.synchronize if job['model'] == 'MENTOR' else lambda: None
    try:
        with baseline_epoch_meter(job['model'], seconds, sync):
            model.fit(split.train_set, split.val_set)
        sync()
        val_tick = time.perf_counter()
        evaluated = evaluate_recommendations(model, split.train_set, split.val_set, ks=(20, 50))
        val_seconds = time.perf_counter() - val_tick
    finally:
        stop.set()
        thread.join(timeout=4)
    if len(seconds) != 3 or not samples:
        raise RuntimeError('Incomplete real epoch or memory measurements')
    elapsed = time.perf_counter() - tick
    ratio = max(1., len(set(split.test_set.uir_tuple[0])) / max(1, len(set(split.val_set.uir_tuple[0]))))
    profile = {'model': job['model'], 'candidate': args.candidate_value, 'batch_size': model.batch_size,
        'parameters': jsonable(model.experiment_params), 'epoch_seconds': seconds,
        'run_overhead_seconds_upper': max(120, elapsed - sum(seconds) + val_seconds * ratio * 1.5),
        'validation_seconds': val_seconds, 'validation_users': evaluated['evaluated_users'],
        'peak_device_memory_MiB': max(samples), 'memory_samples': len(samples),
        'memory_scope': 'device-wide absolute nvidia-smi, not framework allocator or isolated process',
        'max_items': None, 'full_training_catalog': True, 'test_evaluated': False,
        'data_fingerprint': split.preprocessing_report['fingerprint'], 'source_sha256': protocol_sources(),
        'package_versions': baseline_versions(), 'gpu': gpu_identity(), 'elapsed_seconds': elapsed}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / 'profile.json', profile)
    timer.cancel()
    print(json.dumps(profile, ensure_ascii=False, indent=2), flush=True)


def baseline_benchmark(args):
    from experiments.dmcrec.main import write_json
    spec = baseline_spec(args.followup_config)
    commands = []
    for job in baseline_design(spec)['jobs']:
        for value in [.0001, .001]:
            commands.append([sys.executable, '-B', str(Path(__file__).resolve()), 'baseline-measure',
                '--followup-config', str(args.followup_config.resolve()), '--models', job['model'],
                '--candidate-value', str(value), '--cache-dir', str(args.cache_dir)])
    if not args.execute:
        print(json.dumps({'preview': True, 'profiles': 4, 'maximum_total_seconds': 900,
                          'commands': commands}, indent=2))
        return
    price = _positive(args.hourly_price, 'explicit current hourly price')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(args.output_dir):
        receipt = args.output_dir / 'benchmark.json'
        if receipt.exists():
            raise RuntimeError('A completed receipt exists; do not overwrite measurements')
        # Refuse measurement while another application is using the device.
        active = subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid',
                                         '--format=csv,noheader,nounits'], text=True).strip()
        if active:
            raise RuntimeError('Another GPU process is active; refuse mixed device-memory measurements')
        state_path = args.output_dir / 'measurement_state.json'
        state = json.loads(state_path.read_text()) if state_path.exists() else {
            'benchmark_started_at_unix': time.time(), 'hourly_price_CNY': price, 'gpu': gpu_identity()}
        if state['hourly_price_CNY'] != price or state['gpu'] != gpu_identity():
            raise RuntimeError('Price/GPU changed during benchmark attempts')
        start = state['benchmark_started_at_unix']
        state['state'] = 'running'
        write_json(state_path, state)
        profiles = []
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1', 'PYTHONFAULTHANDLER': '1'}
        try:
            for number, command in enumerate(commands):
                directory = args.output_dir / ('profile-' + str(number))
                directory.mkdir(exist_ok=True)
                run_logged_job([*command, '--benchmark-started', str(start), '--output-dir', str(directory)],
                               directory / ('run-' + str(time.time_ns()) + '.log'), environment, start + 900)
                profiles.append(json.loads((directory / 'profile.json').read_text()))
            record = {'schema': 1, 'design_sha256': baseline_digest(spec),
                'source_sha256': protocol_sources(), 'package_versions': baseline_versions(), 'gpu': gpu_identity(),
                'benchmark_started_at_unix': start, 'measured_at_unix': time.time(),
                'hourly_price_CNY': price, 'profiles': profiles}
            projection = baseline_projection(spec, record, 'tune')
            write_json(receipt, record)
            write_json(args.output_dir / 'budget_projection.json', projection)
            state['state'] = 'complete'
            print(json.dumps(projection, indent=2), flush=True)
        except BaseException:
            state['state'] = 'stopped_or_failed'
            raise
        finally:
            write_json(state_path, state)


def baseline_selection(spec, job, directory):
    path, selected = find_selection(directory, 'Baby', 'original', job['model'])
    options = selected.get('training_options', {})
    from experiments.dmcrec.main import source_hashes
    versions = baseline_versions()
    environment = selected.get('environment', {})
    key = next(iter(job['search_grid']))
    if (selected.get('completed_trials') != 2 or selected.get('inference_mode') != 'main'
            or selected.get('environment', {}).get('source_sha256') != source_hashes()
            or environment.get('python') != versions['python']
            or any(environment.get('packages', {}).get(k) != v for k, v in versions.items() if k != 'python')
            or selected.get('data_report', {}).get('fingerprint') != job['data_fingerprint']
            or options.get('search_space') != job['search_grid'] or options.get('search_sampler') != 'grid'
            or options.get('trials') != 2 or options.get('overrides', {}).get(job['model']) != job['fixed_parameters']
            or set(selected.get('best_params', {})) != {key}
            or selected['best_params'][key] not in job['search_grid'][key]
            or options.get('batch_size') != job['batch_size_candidate']
            or options.get('protocol') != 'corrected' or options.get('implementation') != 'corrected'
            or options.get('max_items', 'missing') is not None or options.get('seeds') != [42]
            or any(options.get(k) != v for k, v in spec['common_options'].items())):
        raise RuntimeError('Baseline selection differs from measured fixed protocol/grid/data/code')
    return path, selected


def baseline_campaign(args):
    from experiments.dmcrec.main import write_json
    spec = baseline_spec(args.followup_config)
    if args.stage not in {'tune', 'evaluate'}:
        raise RuntimeError('Use --stage tune or evaluate for the Baby baseline campaign')
    # Config inspection never implies approval; formal work fails before GPU/data access.
    if args.execute and not (spec.get('execution_enabled') is True and spec.get('configuration_frozen') is True
                            and spec['stages']['baby_baselines'].get('budget_approved') is True):
        raise RuntimeError('Baseline configuration/budget not approved and frozen for formal execution')
    jobs = []
    for job in baseline_design(spec)['jobs']:
        flags = baseline_flags(job, spec, args)
        if args.stage == 'tune':
            key = next(iter(job['search_grid']))
            command = [sys.executable, '-B', '-m', 'experiments.dmcrec.main_opt', *flags,
                '--seeds', '42', '--trials', '2', '--search-sampler', 'grid', '--tune-inference', 'main',
                '--search-param', key + '=' + json.dumps(job['search_grid'][key]), '--resume',
                '--output-dir', str(args.output_dir / 'selection')]
            jobs.append((job['model'], command))
        elif args.selection_dir is None:
            raise RuntimeError('Evaluation requires actual new --selection-dir; no guessed best parameters')
        else:
            path, selected = baseline_selection(spec, job, args.selection_dir)
            for key, value in selected['best_params'].items():
                flags += ['--param', job['model'] + '.' + key + '=' + json.dumps(value)]
            for seed in [42, 43, 44, 45, 46]:
                jobs.append((job['model'], [sys.executable, '-B', '-m', 'experiments.dmcrec.main', *flags,
                    '--seeds', str(seed), '--selection-record', str(path), '--resume',
                    '--output-dir', str(args.output_dir / 'evaluate')]))
    if not args.execute:
        for _, command in jobs:
            print(shlex.join(command))
        return
    receipt = args.protocol_record or args.output_dir / 'benchmark.json'
    if not receipt.is_file():
        raise RuntimeError('Fresh four-profile --protocol-record required')
    admitted = baseline_projection(spec, json.loads(receipt.read_text()), args.stage)
    if not admitted['formal_execution_authorized'] or not admitted['within_budget']:
        raise RuntimeError('Baseline queue exceeds shared stage deadline/price cap or lacks authorization')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with campaign_lock(args.output_dir):
        write_json(args.output_dir / ('budget-admission-' + args.stage + '.json'), admitted)
        logs = args.output_dir / 'logs' / args.stage
        logs.mkdir(parents=True, exist_ok=True)
        environment = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1', 'PYTHONUNBUFFERED': '1',
                       'HF_HUB_OFFLINE': '1', 'TRANSFORMERS_OFFLINE': '1'}
        for number, (model, command) in enumerate(jobs):
            run_logged_job(command, logs / (str(time.time_ns()) + '-' + model + '.log'), environment,
                           admitted['deadline_unix'])


def macr_design(path=None):
    """Load the separate MACR proposal without altering the original frozen plan."""
    source = Path(path) if path else ROOT / "artifacts/dmcrec_remote_readiness.json"
    record = json.loads(source.read_text())
    design = record.get("macr_baseline_preparation", record)
    if (design["scope"]["datasets"] != ["Baby", "Sports"]
            or design["scope"]["physical_training_fits"] != 24
            or design["scope"]["seeds"] != [42, 43, 44, 45, 46]
            or design["parameters"]["embedding_dim"] != 64
            or design["parameters"]["batch_size"] != 8192
            or design["parameters"]["learning_rate_candidates"] != [0.0001, 0.001]
            or design["parameters"]["alpha_item_BCE"] != 0.001
            or design["parameters"]["beta_user_BCE"] != 0.001
            or design["parameters"]["sampled_L2"] != 1e-5
            or design["scope"]["reference_grid"] != [0., .5, 1., 2., 5., 10., *map(float, range(20, 41, 2))]
            or design["scope"]["epochs_per_fit"] != 200):
        raise ValueError("MACR design differs from the separate 24-fit proposal")
    expected = record.get("corrected_campaign_analysis", {}).get("data")
    return design, expected


def macr_identity(design, device, threads):
    import platform
    from experiments.dmcrec.main import source_hashes
    protocol = {key: design[key] for key in ["scope", "parameters", "data_and_evaluation"]}
    cpu_model = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                      if line.startswith("model name")), platform.processor()) if Path("/proc/cpuinfo").exists() else platform.processor()
    return {"adapter_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "legacy_sources": source_hashes(), "official_commit": design["official_source"]["commit"],
            "protocol_sha256": hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest(),
            "packages": {k: importlib.metadata.version(k) for k in ["torch", "cornac", "numpy", "scipy"]},
            "python": sys.version, "platform": platform.platform(), "device": device,
            "cpu": {"model": cpu_model, "logical_cpus": os.cpu_count(), "host": platform.node()},
            "threads": threads, "gpu": gpu_identity() if device == "cuda" else None}


def macr_sampler(train_set, batch_size, seed):
    """Official user-uniform sampling, with vectorized positive/negative draws."""
    import numpy as np
    matrix = (train_set.matrix.tocsr() > 0).astype("int8")
    counts = np.diff(matrix.indptr)
    active = np.flatnonzero(counts)
    if batch_size < 1 or not len(active) or (counts[active] >= matrix.shape[1]).any():
        raise ValueError("Sampling requires active users with at least one unobserved item")
    rng = np.random.RandomState(seed)

    def sample():
        users = rng.choice(active, batch_size, replace=batch_size > len(active))
        offsets = (rng.random_sample(batch_size) * counts[users]).astype("int64")
        positive = matrix.indices[matrix.indptr[users] + offsets]
        negative = rng.randint(matrix.shape[1], size=batch_size)
        bad = np.asarray(matrix[users, negative]).ravel() > 0
        while bad.any():
            negative[bad] = rng.randint(matrix.shape[1], size=int(bad.sum()))
            bad = np.asarray(matrix[users, negative]).ravel() > 0
        return users, positive, negative
    return sample


def _macr_score_arrays(arrays, user, mode, reference):
    from scipy.special import expit
    if not 0 <= user < len(arrays["U"]):
        raise IndexError("Unknown user")
    score = arrays["I"] @ arrays["U"][user]
    if mode == "main":
        return score
    if mode not in {"factual", "counterfactual"} or "wi" not in arrays:
        raise ValueError("Modulated scores require a MACR model and a known scoring mode")
    if "item_gate" not in arrays:
        arrays["item_gate"] = expit((arrays["I"] @ arrays["wi"]).ravel())
        arrays["user_gate"] = expit((arrays["U"] @ arrays["wu"]).ravel())
    return (score - (reference if mode == "counterfactual" else 0.0)) * arrays["item_gate"] * arrays["user_gate"][user]


def make_macr_model(n_users, n_items, *, model="MACR-MF", dim=64, seed=42,
                    device="cpu", alpha=0.001, beta=0.001, regularization=1e-5):
    """Independent equation-faithful MF adapter; never registered as DMCRec."""
    import numpy as np
    import torch
    from torch.nn import functional as F
    if model not in {"MACR-MF", "MF-BCE"} or min(n_users, n_items, dim) < 1:
        raise ValueError("Invalid MF adapter model or dimensions")
    if any(not math.isfinite(v) or v < 0 for v in [alpha, beta, regularization]):
        raise ValueError("Regularization weights must be nonnegative")

    class Adapter(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.model, self.seed = model, seed
            self.mode = "counterfactual" if model == "MACR-MF" else "main"
            self.reference = 0.0
            self.history, self._arrays, self.fit_options = [], None, None
            generator = torch.Generator(device=device).manual_seed(seed)
            shapes = {"U": (n_users, dim), "I": (n_items, dim)}
            if model == "MACR-MF":
                shapes.update(wi=(dim, 1), wu=(dim, 1))
            for name, shape in shapes.items():
                value = torch.empty(shape, dtype=torch.float32, device=device)
                bound = math.sqrt(6 / sum(shape))
                value.uniform_(-bound, bound, generator=generator)
                self.register_parameter(name, torch.nn.Parameter(value))

        def loss(self, users, positive, negative):
            u, p, n = self.U[users], self.I[positive], self.I[negative]
            ps, ns = (u * p).sum(1), (u * n).sum(1)
            auxiliary = 0.0
            if self.model == "MACR-MF":
                ui, pi, ni = (u @ self.wu).squeeze(1), (p @ self.wi).squeeze(1), (n @ self.wi).squeeze(1)
                ps = ps * torch.sigmoid(ui) * torch.sigmoid(pi)
                ns = ns * torch.sigmoid(ui) * torch.sigmoid(ni)
                auxiliary = alpha * (F.softplus(-pi) + F.softplus(ni)).mean()
                auxiliary = auxiliary + beta * (F.softplus(-ui) + F.softplus(ui)).mean()
            if ps.shape != users.shape or ns.shape != users.shape:
                raise RuntimeError("Joint scores must be one scalar per sample, never B-by-B")
            joint = (F.softplus(-ps) + F.softplus(ns)).mean()
            penalty = regularization * (u.square().sum() + p.square().sum() + n.square().sum()) / (2 * len(users))
            return joint + auxiliary + penalty

        def fit(self, train_set, *, epochs=200, batch_size=8192, learning_rate=0.001, deadline=None):
            if epochs < 1 or batch_size < 1 or not math.isfinite(learning_rate) or learning_rate <= 0:
                raise ValueError("Positive epochs, batch size and finite learning rate required")
            if (train_set.num_users, train_set.num_items) != (n_users, n_items):
                raise ValueError("Training catalog differs from the adapter")
            sample = macr_sampler(train_set, batch_size, self.seed)
            optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, eps=1e-8)
            batches = train_set.matrix.nnz // batch_size + 1
            self.history, self._arrays, self.fit_options = [], None, None
            for epoch in range(1, epochs + 1):
                started = time.perf_counter()
                losses = []
                for _ in range(batches):
                    if deadline is not None and time.monotonic() >= deadline:
                        raise TimeoutError("MF adapter training deadline")
                    batch = [torch.as_tensor(a, dtype=torch.long, device=device) for a in sample()]
                    optimizer.zero_grad(set_to_none=True)
                    loss = self.loss(*batch)
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Nonfinite MF adapter loss")
                    loss.backward()
                    optimizer.step()
                    losses.append(float(loss.detach()))
                if device == "cuda":
                    torch.cuda.synchronize()
                self.history.append({"epoch": epoch, "seconds": time.perf_counter() - started,
                                     "loss": float(np.mean(losses)), "batches": batches})
            self._arrays = {k: v.detach().cpu().numpy().copy() for k, v in self.named_parameters()}
            self.fit_options = {"epochs": epochs, "batch_size": batch_size, "learning_rate": learning_rate,
                                "sampler": "user_uniform", "batches_per_epoch": batches, "optimizer": "Adam", "eps": 1e-8}
            return self

        def score(self, user):
            if self._arrays is None:
                self._arrays = {k: v.detach().cpu().numpy().copy() for k, v in self.named_parameters()}
            return _macr_score_arrays(self._arrays, user, self.mode, self.reference)

        def export(self, path, train_set, metadata):
            # Unicode IDs and a JSON scalar keep every field readable without pickle/GPU.
            if self.fit_options is None:
                raise ValueError("Only a completed fit can be exported as a checkpoint")
            def ids(mapping, size):
                result = [None] * size
                for key, index in mapping.items():
                    if index < size:
                        result[index] = str(key)
                if any(v is None for v in result) or len(set(result)) != size:
                    raise ValueError("Incomplete or nonunique training ID map")
                return np.asarray(result, dtype=str)
            info = {"schema": 1, "model": self.model, "seed": self.seed, "dim": dim,
                    "num_users": n_users, "num_items": n_items,
                    "mode": self.mode, "reference": self.reference, "history": self.history,
                    "alpha_item": alpha, "beta_user": beta, "regularization": regularization,
                    "fit_options": self.fit_options, "provenance": metadata}
            arrays = {k: v.detach().cpu().numpy() for k, v in self.named_parameters()}
            arrays.update(user_ids=ids(train_set.uid_map, n_users), item_ids=ids(train_set.iid_map, n_items),
                          metadata=np.asarray(json.dumps(info, sort_keys=True, allow_nan=False)))
            path = Path(path)
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(path.suffix + ".tmp")
            with temporary.open("wb") as handle:
                np.savez_compressed(handle, **arrays)
            temporary.replace(path)
            return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    return Adapter()


def load_macr_portable(path, expected_provenance=None):
    import numpy as np
    with np.load(path, allow_pickle=False) as file:
        info = json.loads(file["metadata"].item())
        if info.get("schema") != 1 or info.get("model") not in {"MACR-MF", "MF-BCE"}:
            raise ValueError("Unknown portable MF schema/model")
        keys = ["U", "I"] + (["wi", "wu"] if info["model"] == "MACR-MF" else [])
        arrays = {key: file[key].copy() for key in keys}
        if expected_provenance is not None and info["provenance"] != expected_provenance:
            raise ValueError("Portable source/data/selection provenance mismatch")
        if (arrays["U"].ndim != 2 or arrays["I"].ndim != 2
                or arrays["U"].shape[1] != info["dim"] or arrays["I"].shape[1] != info["dim"]
                or len(arrays["U"]) != info["num_users"] or len(arrays["I"]) != info["num_items"]
                or len(file["user_ids"]) != len(arrays["U"]) or len(file["item_ids"]) != len(arrays["I"])):
            raise ValueError("Portable catalog/embedding shape mismatch")
        if any(not np.isfinite(value).all() for value in arrays.values()):
            raise ValueError("Nonfinite portable MF parameters")
        if "wi" in arrays and any(arrays[key].shape != (info["dim"], 1) for key in ["wi", "wu"]):
            raise ValueError("Portable gate shape mismatch")

    class Portable:
        mode, reference, metadata = info["mode"], info["reference"], info

        def score(self, user):
            return _macr_score_arrays(arrays, user, self.mode, self.reference)
    return Portable()


def macr_evaluate(model, train_set, heldout_set, *, exclude_set=None, ks=(20, 50), deadline=None):
    from experiments.dmcrec.exutils import evaluate_recommendations

    class TimedScore:
        def score(self, user):
            if deadline is not None and time.monotonic() >= deadline:
                raise TimeoutError("MF adapter evaluation deadline")
            return model.score(user)
    return evaluate_recommendations(TimedScore(), train_set, heldout_set, ks=ks, exclude_set=exclude_set)


def macr_select_reference(model, train_set, validation_set, references, deadline=None):
    if validation_set is None or not references or any(not math.isfinite(c) for c in references):
        raise ValueError("Finite references and an independent validation split required")
    model.mode = "counterfactual" if model.model == "MACR-MF" else "main"
    trials = []
    for reference in (sorted(set(references)) if model.model == "MACR-MF" else [0.0]):
        model.reference = reference
        result = macr_evaluate(model, train_set, validation_set, ks=(20,), deadline=deadline)
        trials.append({"reference": reference, "validation_ndcg20": result["metrics"]["NDCG@20"]})
    best = max(trials, key=lambda t: (t["validation_ndcg20"], -t["reference"]))
    model.reference = best["reference"]
    return {"selection_split": "validation", "test_evaluated": False, "trials": trials, **best}


def macr_profile_projection(design, profiles):
    expected = {(d, m, lr) for d in ["Baby", "Sports"] for m in ["MF-BCE", "MACR-MF"] for lr in [0.0001, 0.001]}
    keys = [(p["dataset"], p["model"], p["learning_rate"]) for p in profiles]
    if set(keys) != expected or len(keys) != 8:
        raise ValueError("All eight unique full-catalog MACR/MF profiles are required")
    training = validation = final = 0.0
    for dataset in ["Baby", "Sports"]:
        for model in ["MF-BCE", "MACR-MF"]:
            rows = [p for p in profiles if p["dataset"] == dataset and p["model"] == model]
            if any(len(p["epochs"]) != 3 or p["scope"] != "full_catalog_resource_only" for p in rows):
                raise ValueError("Three real full-catalog epochs are required per profile")
            epoch = max(v["seconds"] for p in rows for v in p["epochs"])
            val = max(p["validation_seconds"] for p in rows)
            training += 6 * (200 * epoch * 1.5 + 120)
            validation += 2 * (17 if model == "MACR-MF" else 1) * val * 1.5
            # The probe evaluates only @20; final reports evaluate @20 AND @50.
            final += 2 * 5 * (3 if model == "MACR-MF" else 1) * max(p["test_users_expected"] / p["validation_users"] for p in rows) * val * 1.5
    seconds = training + validation + final + 900
    return {"physical_fits": 24, "final_seed_results": 20, "projected_remaining_upper_seconds": seconds,
            "training_upper_seconds": training, "validation_upper_seconds": validation,
            "final_scoring_upper_seconds": final, "export_archive_reserve_seconds": 900,
            "validation_cutoffs": [20], "final_cutoffs": [20, 50], "final_metric_cost_multiplier": 2,
            "within_proposed_four_hours_with_20pct_reserve": seconds <= 4 * 3600 * 0.8,
            "GPU_time_or_cost_estimate": False, "formal_execution_authorized": False}


def macr_checkpoint_selection(paths, design, identity, data_report):
    """Bind two validation candidates and reuse the selected seed42 checkpoint."""
    candidates = []
    model_name = None
    for path in paths:
        portable = load_macr_portable(path)
        meta = portable.metadata
        provenance, options = meta["provenance"], meta["fit_options"]
        if (provenance.get("identity") != identity or provenance.get("data") != data_report
                or provenance.get("scope") != "validation_candidate" or meta["seed"] != 42
                or meta["dim"] != design["parameters"]["embedding_dim"]
                or options["epochs"] != design["scope"]["epochs_per_fit"]
                or options["batch_size"] != design["parameters"]["batch_size"]
                or meta["alpha_item"] != design["parameters"]["alpha_item_BCE"]
                or meta["beta_user"] != design["parameters"]["beta_user_BCE"]
                or meta["regularization"] != design["parameters"]["sampled_L2"]
                or options["sampler"] != "user_uniform" or options["optimizer"] != "Adam" or options["eps"] != 1e-8
                or len(meta["history"]) != options["epochs"]
                or [h["epoch"] for h in meta["history"]] != list(range(1, options["epochs"] + 1))):
            raise ValueError("Candidate checkpoint training/source/data identity mismatch")
        model_name = model_name or meta["model"]
        if meta["model"] != model_name:
            raise ValueError("Cannot select across different MF models")
        if meta["mode"] != ("counterfactual" if model_name == "MACR-MF" else "main"):
            raise ValueError("Candidate scoring mode differs from the prespecified primary score")
        catalog = data_report.get("train_after_filter")
        if catalog and (meta["num_users"] != catalog["users"] or meta["num_items"] != catalog["items"]):
            raise ValueError("Candidate catalog differs from the data report")
        choice = provenance.get("validation_selection", {})
        references = design["scope"]["reference_grid"] if model_name == "MACR-MF" else [0.0]
        trials = choice.get("trials", [])
        if (choice.get("selection_split") != "validation" or choice.get("test_evaluated") is not False
                or sorted(t["reference"] for t in trials) != sorted(references)
                or len(trials) != len(set(references))
                or any(not math.isfinite(t["validation_ndcg20"]) for t in trials)):
            raise ValueError("Candidate reference search must be the fixed validation-only grid")
        best = max(trials, key=lambda t: (t["validation_ndcg20"], -t["reference"]))
        if (choice["reference"] != best["reference"] or choice["validation_ndcg20"] != best["validation_ndcg20"]
                or meta["reference"] != best["reference"]):
            raise ValueError("Checkpoint reference differs from validation selection")
        candidates.append({"path": str(Path(path).resolve()), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                           "learning_rate": options["learning_rate"], **best})
    if sorted(c["learning_rate"] for c in candidates) != sorted(design["parameters"]["learning_rate_candidates"]):
        raise ValueError("Exactly the two prespecified learning-rate candidates are required")
    winner = max(candidates, key=lambda c: (c["validation_ndcg20"], -c["reference"], -c["learning_rate"]))
    return {"schema": 1, "identity": identity, "data": data_report, "model": model_name,
            "selection_split": "validation", "test_evaluated": False, "candidates": candidates,
            "winner": winner, "reuse_seed42_without_retraining": True}


def macr_reuse_winner(selection, design, identity, data_report):
    if (selection.get("identity") != identity or selection.get("data") != data_report
            or selection.get("selection_split") != "validation" or selection.get("test_evaluated") is not False):
        raise ValueError("Selection source/data/validation identity mismatch")
    paths = [c["path"] for c in selection["candidates"]]
    if any(hashlib.sha256(Path(c["path"]).read_bytes()).hexdigest() != c["sha256"] for c in selection["candidates"]):
        raise ValueError("Selected candidate bytes changed")
    verified = macr_checkpoint_selection(paths, design, identity, data_report)
    if verified != selection:
        raise ValueError("Selection winner or candidate record changed")
    return load_macr_portable(selection["winner"]["path"])


def macr_profile(args):
    """Bounded resource-only runs, never evaluating the project test split."""
    from experiments.dmcrec.main import parse_args, load_data, write_json
    design, expected = macr_design(args.followup_config)
    if not args.execute:
        print(json.dumps({"action": "macr-profile", "device": args.device, "epochs": 3,
                          "datasets": args.datasets or ["Baby", "Sports"],
                          "models": args.models or ["MF-BCE", "MACR-MF"],
                          "learning_rates": [0.0001, 0.001], "test_evaluated": False}, indent=2))
        return
    import gc
    import torch
    from threadpoolctl import threadpool_limits
    datasets, models = args.datasets or ["Baby", "Sports"], args.models or ["MF-BCE", "MACR-MF"]
    if set(datasets) - {"Baby", "Sports"} or set(models) - {"MF-BCE", "MACR-MF"}:
        raise ValueError("Only the separate MACR/MF proposal may be profiled")
    if expected is None:
        raise ValueError("Existing corrected data reports are required to bind the resource measurements")
    torch.set_num_threads(args.threads)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested CUDA device unavailable")
    started = time.monotonic()
    deadline = started + args.profile_seconds
    identity = macr_identity(design, args.device, args.threads)
    receipt = {"identity": identity, "profiles": [], "state": "running", "test_evaluated": False}
    output = args.output_dir / "macr-profile.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, receipt)
    with threadpool_limits(limits=args.threads):
        try:
            for dataset in datasets:
                options = parse_args(["--dataset", dataset, "--device", args.device,
                    "--epochs", "3", "--batch-size", "8192", "--setting", "original", "--protocol", "corrected",
                    "--split-seed", "42", "--val-size", "0.1", "--test-size", "0.2", "--quiet",
                    "--cache-dir", str(args.cache_dir)])
                split, features = load_data(options, "original")
                report = split.preprocessing_report
                for key in ["fingerprint", "features_sha256", "train_after_filter", "validation_known", "test_known"]:
                    if report.get(key) != expected[dataset].get(key):
                        raise RuntimeError("MACR profile data differs from completed corrected study: " + key)
                print(json.dumps({"loaded_dataset": dataset, "fingerprint": report["fingerprint"]}), flush=True)
                for model in models:
                    for lr in [0.0001, 0.001]:
                        if macr_identity(design, args.device, args.threads) != identity:
                            raise RuntimeError("MACR adapter/source/environment changed during measurement")
                        fitted = make_macr_model(split.train_set.num_users, split.train_set.num_items,
                                                 model=model, device=args.device)
                        fitted.fit(split.train_set, epochs=3, learning_rate=lr, deadline=deadline)
                        fitted.mode, fitted.reference = ("counterfactual" if model == "MACR-MF" else "main"), 0.0
                        measured = macr_evaluate(fitted, split.train_set, split.val_set, ks=(20,), deadline=deadline)
                        # Resource validation scores are deliberately not retained or selected.
                        p = {"dataset": dataset, "model": model, "learning_rate": lr,
                             "scope": "full_catalog_resource_only", "epochs": fitted.history,
                             "validation_seconds": measured["seconds"], "validation_users": measured["evaluated_users"],
                             "test_users_expected": {"Baby": 15253, "Sports": 27836}[dataset],
                             "data_report": report, "test_evaluated": False,
                             "parameter_count": sum(v.numel() for v in fitted.parameters())}
                        receipt["profiles"].append(p)
                        receipt["elapsed_seconds"] = time.monotonic() - started
                        write_json(output, receipt)
                        print(json.dumps({"dataset": dataset, "model": model, "lr": lr,
                            "epoch_seconds": [h["seconds"] for h in fitted.history],
                            "validation_seconds": measured["seconds"]}), flush=True)
                        del fitted, measured
                        gc.collect()
                del split, features
                gc.collect()
            receipt["state"] = "complete"
            if len(receipt["profiles"]) == 8:
                receipt["projection"] = macr_profile_projection(design, receipt["profiles"])
        except BaseException as exc:
            receipt["state"], receipt["error"] = "stopped_or_failed", str(exc)
            raise
        finally:
            receipt["elapsed_seconds"] = time.monotonic() - started
            write_json(output, receipt)


def make_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["protocol", "preflight", "smoke", "prepare-text", "campaign", "collect", "baseline-benchmark", "baseline-measure", "baseline-campaign", "baseline-budget", "macr-profile"])
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--plan", type=Path, default=PROJECT / "remote_plan.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "dmcrec_results/focused_v2")
    parser.add_argument("--protocol-record", type=Path, help="Real GPU timing/environment record required for frozen formal execution")
    parser.add_argument("--cache-dir", type=Path, default=PROJECT / "cache")
    parser.add_argument("--hf-endpoint", default=None)
    parser.add_argument("--allow-legacy-text-cache", action="store_true", help="Exploratory runs only; positional cache provenance remains unverified")
    parser.add_argument("--stage", choices=["pilot", "tune", "evaluate", "ablate", "refresh"], default="pilot")
    parser.add_argument("--datasets", nargs="+")
    parser.add_argument("--models", nargs="+")
    parser.add_argument("--settings", nargs="+", choices=["original", "sparse"])
    parser.add_argument("--selection-dir", type=Path)
    parser.add_argument("--use-presets", action="store_true")
    parser.add_argument("--save-models", action="store_true")
    parser.add_argument("--verify-hashes", action="store_true")
    parser.add_argument("--execute", action="store_true", help="Actually execute campaign commands; default is preview only")
    parser.add_argument("--followup-config", type=Path, help="Separate follow-up design JSON or existing readiness JSON")
    parser.add_argument("--hourly-price", type=float, help="Explicit current price for a new measured stage")
    parser.add_argument("--candidate-value", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--benchmark-started", type=float, help=argparse.SUPPRESS)
    parser.add_argument("--threads", type=int, choices=[1, 2], default=1, help="CPU threads for independent MACR measurements")
    parser.add_argument("--profile-seconds", type=int, default=300, help="Whole MACR resource measurement deadline")
    return parser


if __name__ == "__main__":
    args = make_parser().parse_args()
    if args.action == "macr-profile":
        if args.output_dir == ROOT / "dmcrec_results/focused_v2":
            raise ValueError("MACR measurements require an explicit separate --output-dir")
        if args.profile_seconds < 1:
            raise ValueError("Positive MACR profiling time limit required")
        macr_profile(args)
    elif args.action.startswith("baseline-"):
        if args.output_dir == ROOT / "dmcrec_results/focused_v2":
            args.output_dir = ROOT / "dmcrec_results/followup_v3/baby_baselines"
        if args.action == "baseline-measure":
            baseline_measure(args)
        elif args.action == "baseline-benchmark":
            baseline_benchmark(args)
        elif args.action == "baseline-campaign":
            baseline_campaign(args)
        else:
            record = args.protocol_record or args.output_dir / "benchmark.json"
            print(json.dumps(baseline_projection(baseline_spec(args.followup_config),
                              json.loads(record.read_text()), args.stage), indent=2))
    elif args.action == "protocol":
        print(json.dumps(protocol_summary(read_plan(args.plan)), ensure_ascii=False, indent=2))
    elif args.action == "preflight":
        preflight(args)
    elif args.action == "prepare-text":
        prepare_text(args)
    elif args.action == "collect":
        collect(args)
    elif args.action == "smoke":
        environment = {**os.environ, "DMCREC_TEST_DEVICE": args.device, "PYTHONDONTWRITEBYTECODE": "1",
                       "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
        subprocess.run([sys.executable, "-B", "-m", "unittest", "discover", "-s", "tests", "-p", "test_dmcrec_remote.py", "-v"],
                       cwd=ROOT, env=environment, check=True)
    else:
        jobs = planned_jobs(args)
        print(f"Stage: {args.stage}; jobs: {len(jobs)}; execute: {args.execute}")
        if not args.execute:
            for dataset, setting, model, variant, command in jobs:
                print(shlex.join(command))
        else:
            with campaign_lock(args.output_dir):
                execute(args, jobs)
