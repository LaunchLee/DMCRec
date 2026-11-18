import os, pickle, random, torch
import warnings
import numpy as np
import pandas as pd
import joblib, random
from collections import Counter
from cornac.data import TextModality, ImageModality
from cornac.eval_methods import BaseMethod, RatioSplit


THIS_DIR = os.path.dirname(os.path.abspath(__file__))


def create_dir_if_not_exist(*dirname: str) -> str:
    the_dir = os.path.abspath(os.path.join(THIS_DIR, *dirname))
    os.makedirs(the_dir, exist_ok=True)
    return the_dir


def seed_everything(seed=42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'


def encode_texts(texts, name, cache_dir="cache", model_name="intfloat/e5-base", batch_size=256, verbose=False, ids=None):
    warnings.filterwarnings("ignore", module="transformers")
    warnings.filterwarnings("ignore", module="huggingface_hub")

    import hashlib, json
    ids = list(map(str, ids)) if ids is not None else [str(i) for i in range(len(texts))]
    if len(ids) != len(texts) or len(set(ids)) != len(ids):
        raise ValueError("Text IDs must be unique and match text rows")
    signature = hashlib.sha256()
    for item_id, text in sorted(zip(ids, texts)):
        signature.update(json.dumps([item_id, text], ensure_ascii=False).encode())
        signature.update(b"\n")
    text_digest = signature.hexdigest()
    embedding_dir = os.path.abspath(os.path.join(cache_dir, "_embeddings"))
    save_path = os.path.join(embedding_dir, f"{name}_embeddings.joblib")
    if os.path.exists(save_path):
        if verbose:
            print(f"[INFO] Reusing text embeddings from {save_path}")
        cached = joblib.load(save_path)
        if isinstance(cached, dict):
            if cached.get("model") != model_name or cached.get("text_sha256") != text_digest:
                raise ValueError(f"Text cache model/content mismatch: {save_path}; use another cache directory or explicitly regenerate it")
            cached_ids = cached.get("ids", [])
            embeddings = np.asarray(cached.get("embeddings"))
            if len(cached_ids) != len(set(cached_ids)) or set(cached_ids) != set(ids):
                raise ValueError(f"Text cache item IDs mismatch: {save_path}")
            if embeddings.ndim != 2 or len(embeddings) != len(cached_ids):
                raise ValueError(f"Text cache feature rows mismatch: {save_path}")
            positions = {item_id: i for i, item_id in enumerate(cached_ids)}
            embeddings = embeddings[[positions[item_id] for item_id in ids]]
        else:
            embeddings = np.asarray(cached)
            from pathlib import Path
            mapping_path = Path(save_path).with_suffix(".ids.json")
            if mapping_path.is_file():
                metadata = json.loads(mapping_path.read_text(encoding="utf-8"))
                cache_digest = hashlib.sha256()
                with open(save_path, "rb") as handle:
                    for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                        cache_digest.update(block)
                cached_ids = metadata.get("ids", [])
                if (metadata.get("model") != model_name or metadata.get("text_sha256") != text_digest
                        or metadata.get("cache_sha256") != cache_digest.hexdigest()
                        or len(cached_ids) != len(set(cached_ids)) or set(cached_ids) != set(ids)):
                    raise ValueError(f"Text cache row-ID mapping mismatch: {mapping_path}")
                if embeddings.ndim != 2 or len(embeddings) != len(cached_ids):
                    raise ValueError(f"Text cache row count mismatch: {mapping_path}")
                positions = {item_id: i for i, item_id in enumerate(cached_ids)}
                embeddings = embeddings[[positions[item_id] for item_id in ids]]
            else:
                warnings.warn(f"Legacy text cache has no row-ID mapping; reusing original encoding order: {save_path}", RuntimeWarning)
        if embeddings.ndim != 2 or len(embeddings) != len(ids) or not np.isfinite(embeddings).all():
            raise ValueError(f"Invalid text cache shape/values: {save_path}")
        return embeddings

    from sentence_transformers import SentenceTransformer

    if "e5-" in model_name.lower():
        processed_texts = [
            t if t.startswith("passage: ") else f"passage: {t}" for t in texts
        ]
    else:
        processed_texts = texts

    model = SentenceTransformer(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu" if 'torch' in globals() else None
    embeddings = model.encode(
        processed_texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        show_progress_bar=True,
        device=device
    )
    os.makedirs(embedding_dir, exist_ok=True)
    import tempfile
    descriptor, temporary = tempfile.mkstemp(prefix=".text-encoding-", suffix=".joblib", dir=embedding_dir)
    os.close(descriptor)
    try:
        joblib.dump({"ids": ids, "model": model_name, "text_sha256": text_digest,
                     "embeddings": embeddings}, temporary)
        os.replace(temporary, save_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return embeddings


def multimodalize(texts, text_ids, visuals, visual_ids):
    if len(texts) != len(text_ids) or len(visuals) != len(visual_ids):
        raise ValueError("Feature row counts must match their item ID lists")
    if len(set(text_ids)) != len(text_ids) or len(set(visual_ids)) != len(visual_ids):
        raise ValueError("Duplicate item IDs in multimodal features")
    text_id_to_idx = {uid: idx for idx, uid in enumerate(text_ids)}

    mm_ids = []
    mm_texts = []
    mm_visuals = []

    for i, v_idx in enumerate(visual_ids):
        if v_idx in text_id_to_idx:
            t_idx = text_id_to_idx[v_idx]
            mm_ids.append(v_idx)
            mm_texts.append(texts[t_idx])
            mm_visuals.append(visuals[i])

    if not mm_ids:
        raise ValueError("No overlapping visual/text item IDs")
    return np.array(mm_ids), np.asarray(mm_texts, dtype=np.float32), np.asarray(mm_visuals, dtype=np.float32)


def get_latest_pkl(dir_path: str):
    if not os.path.exists(dir_path):
        return None

    pkl_files = [
        os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".pkl")
    ]

    if not pkl_files:
        return None

    return max(pkl_files, key=os.path.getmtime)


def load_recent_model(dir_name: str, name: str):
    target_dir = os.path.join(dir_name, name)
    if not os.path.exists(target_dir):
        print(f"[Warning] Cannot find {target_dir}")
        return None
    file_path = get_latest_pkl(target_dir)
    if file_path is None:
        print(f"[Warning] Cannot find any valid .pkl model in {target_dir}")
        return None
    with open(file_path, 'rb') as f:
        return pickle.load(f)


def make_dataset_long_tail(feedback_data, alpha=0.5, target_count=50, min_user_inter=2, seed=None, verbose=False):
    rng = random.Random(seed) if seed is not None else random.Random()

    item_counts = Counter([item[1] for item in feedback_data])
    long_tail_feedback = []

    for user_id, item_id, rating in feedback_data:
        count = item_counts[item_id]

        if count > target_count:
            keep_prob = (target_count / count) ** alpha
        else:
            keep_prob = 1.0

        if rng.random() < keep_prob:
            long_tail_feedback.append((user_id, item_id, rating))

    user_counts = Counter([item[0] for item in long_tail_feedback])
    final_feedback = [
        item for item in long_tail_feedback
        if user_counts[item[0]] >= min_user_inter
    ]

    if verbose:
        original_size = len(feedback_data)
        new_size = len(final_feedback)
        keep_rate = new_size / original_size if original_size > 0 else 0

        new_item_counts = Counter([item[1] for item in final_feedback])
        new_max_count = max(new_item_counts.values()) if new_item_counts else 0

        print(f"[Long-Tail Filter] Original edges: {original_size}, items: {len(item_counts)}")
        print(f"[Long-Tail Filter]   Filtered edges: {new_size} (Keep rate: {keep_rate:.2%})")
        print(f"[Long-Tail Filter]   Filtered items: {len(new_item_counts)}")
        print(f"[Long-Tail Filter]   Max trendings: {max(item_counts.values())} -> {new_max_count}")
        print("-" * 79)

    return final_feedback


def _idx2raw_dict(contents, raw_ids, iid_map: dict):
    raw_id_to_content = {raw_id: content for content, raw_id in zip(contents, raw_ids)}
    id_mapping_dict = {int_idx: raw_id for raw_id, int_idx in iid_map.items()}
    item_idx_to_raw_content = {}
    for int_idx, raw_id in id_mapping_dict.items():
        if raw_id in raw_id_to_content:
            item_idx_to_raw_content[int_idx] = raw_id_to_content[raw_id]
        else:
            item_idx_to_raw_content[int_idx] = f"Item_ID_{raw_id}"
    return item_idx_to_raw_content


def apply_k_core(feedback_data, k_core=5, verbose=False):
    if k_core is None or k_core <= 1:
        return feedback_data

    df = pd.DataFrame(feedback_data, columns=["user", "item", "rating"])

    start_len = len(df)
    while True:
        current_len = len(df)
        user_counts = df["user"].value_counts()
        item_counts = df["item"].value_counts()

        df = df[df["user"].isin(user_counts[user_counts >= k_core].index)]
        df = df[df["item"].isin(item_counts[item_counts >= k_core].index)]

        if len(df) == current_len:
            break

    if verbose:
        print(f"[K-Core Filter] Applied {k_core}-core filter: ")
        print(f"[K-Core Filter]   Interacts {start_len} -> {len(df)}")
        print(f"[K-Core Filter]   Users: {df['user'].nunique()}")
        print(f"[K-Core Filter]   Items: {df['item'].nunique()}")
        print('-' * 79)

    return list(df.itertuples(index=False, name=None))


def dataset_preprocessing(name, visuals, visual_ids, texts, text_ids, feedback, *, texts_embeded=False, raw_texts=None, text_enc_model="intfloat/e5-base", cache_dir="cache", batch_size=256, max_samples=None, make_long_tail=True, k_core=None, seed=None, verbose=False, protocol="corrected", val_size=0.1, test_size=0.2, provided_splits=None, source_metadata=None):
    if protocol not in {"corrected", "legacy"}:
        raise ValueError("Unknown data protocol")
    if not 0 < test_size < 1 or not 0 <= val_size < 1 or val_size + test_size >= 1:
        raise ValueError("Invalid validation/test fractions")
    if verbose:
        print('-' * 79)

    if provided_splits is not None and protocol != "corrected":
        raise ValueError("Provided splits require the corrected protocol")
    text_embedings = texts if texts_embeded else encode_texts(texts, f"{name}_text", model_name=text_enc_model, cache_dir=cache_dir, batch_size=batch_size, verbose=verbose, ids=text_ids)

    mm_ids, mm_text_embedings, mm_visuals = multimodalize(
        text_embedings,
        text_ids,
        visuals,
        visual_ids
    )

    if max_samples is not None and max_samples < len(mm_ids):
        if verbose:
            print(f"[INFO] Sampling items: {len(mm_ids)} -> {max_samples}")
        rng = random.Random(seed)
        sampled_indices = rng.sample(range(len(mm_ids)), max_samples)
        sampled_indices.sort()
        mm_ids = [mm_ids[i] for i in sampled_indices]
        if isinstance(mm_text_embedings, np.ndarray):
            mm_text_embedings = mm_text_embedings[sampled_indices]
        else:
            mm_text_embedings = [mm_text_embedings[i] for i in sampled_indices]
        if isinstance(mm_visuals, np.ndarray):
            mm_visuals = mm_visuals[sampled_indices]
        else:
            mm_visuals = [mm_visuals[i] for i in sampled_indices]

    valid_item_sets = set(mm_ids)
    def usable(rows):
        output = []
        for u, i, rating in rows:
            if not np.isfinite(rating):
                raise ValueError("Nonfinite interaction rating")
            if i in valid_item_sets and (protocol == "legacy" or rating >= 1):
                output.append((u, i, 1.0 if protocol == "corrected" else rating))
        return output

    filtered_feedback = usable(feedback)

    before_stats = feedback_statistics(filtered_feedback)
    if protocol == "corrected":
        # Keep each user-item pair in one split, even for repeated events.
        filtered_feedback = list({(u, i): (u, i, r) for u, i, r in filtered_feedback}.values())
        if provided_splits is None:
            train_data, val_data, test_data = split_feedback(filtered_feedback, val_size, test_size, seed)
        else:
            if len(provided_splits) != 3:
                raise ValueError("Expected provided train/validation/test records")
            train_data, val_data, test_data = [list({(u, i): (u, i, r) for u, i, r in usable(rows or [])}.values())
                                              for rows in provided_splits]
            seen = set()
            for part in [train_data, val_data, test_data]:
                pairs = {(u, i) for u, i, _ in part}
                if seen & pairs:
                    raise ValueError("User-item pairs overlap between provided splits")
                seen.update(pairs)
            if seen != {(u, i) for u, i, _ in filtered_feedback}:
                raise ValueError("Provided splits do not cover the available interactions")
            if not train_data or not test_data:
                raise ValueError("Provided train/test split is empty after feature filtering")
            val_data = val_data or None
        train_before = feedback_statistics(train_data)
        if make_long_tail:
            train_data = make_dataset_long_tail(train_data, seed=seed, verbose=verbose)
        if k_core is not None:
            train_data = apply_k_core(train_data, k_core=k_core, verbose=verbose)
        train_after = feedback_statistics(train_data)
    elif make_long_tail:
        filtered_feedback = make_dataset_long_tail(
            feedback_data=filtered_feedback,
            seed=seed,
            verbose=verbose
        )

    if protocol == "legacy" and k_core is not None:
        filtered_feedback = apply_k_core(filtered_feedback, k_core=k_core, verbose=verbose)

    filtered_items = set(item[1] for item in filtered_feedback)
    keep_indices = [i for i, item_id in enumerate(mm_ids) if item_id in filtered_items]

    mm_ids = [mm_ids[i] for i in keep_indices]
    if isinstance(mm_text_embedings, np.ndarray):
        mm_text_embedings = mm_text_embedings[keep_indices]
    else:
        mm_text_embedings = [mm_text_embedings[i] for i in keep_indices]

    if isinstance(mm_visuals, np.ndarray):
        mm_visuals = mm_visuals[keep_indices]
    else:
        mm_visuals = [mm_visuals[i] for i in keep_indices]

    # --- (Sparsity & Gini Index) ---
    num_interacts = len(filtered_feedback)
    num_users = len(set(item[0] for item in filtered_feedback))
    num_items = len(mm_ids)

    # 1. (Sparsity)
    if num_users > 0 and num_items > 0:
        sparsity = 1.0 - (num_interacts / (num_users * num_items))
    else:
        sparsity = 1.0

    # 2. (Gini Index)
    item_counts = Counter(item[1] for item in filtered_feedback)
    y = np.sort(list(item_counts.values()))
    N = len(y)
    if N > 0 and np.sum(y) > 0:
        index = np.arange(1, N + 1)
        gini_index = (2 * np.sum(index * y) / (N * np.sum(y))) - (N + 1) / N
    else:
        gini_index = 0.0

    if verbose:
        print("[INFO] Dataset name:", name)
        print("[INFO]   Original interacts:", len(feedback))
        print("[INFO]   Filtered interacts:", len(filtered_feedback))
        print("[INFO]   Users:", len(set(item[0] for item in filtered_feedback)))
        print("[INFO]   Items:", len(mm_ids))
        print(f"[INFO]   Sparsity: {sparsity * 100:.4f}%")
        print(f"[INFO]   Gini Index: {gini_index:.4f}")
        print('-' * 79)

    # modals ---------------------------------------------------------------------
    text_modality = TextModality(
        features=mm_text_embedings,
        ids=mm_ids  # type: ignore
    )

    visual_modality = ImageModality(
        features=mm_visuals,
        ids=mm_ids,
        normalized=True
    )

    options = dict(
        rating_threshold=1.0,
        item_text=text_modality,
        item_image=visual_modality,
        exclude_unknowns=True,
        verbose=verbose,
        seed=seed
    )
    if protocol == "corrected":
        ratio_split = BaseMethod.from_splits(train_data=train_data, val_data=val_data,
                                            test_data=test_data, **options)
        import hashlib, json
        fingerprint = hashlib.sha256(json.dumps([train_data, val_data, test_data],
                                               ensure_ascii=False).encode()).hexdigest()
        known_users = {r[0] for r in train_data}
        known_items = {r[1] for r in train_data}
        ratio_split.preprocessing_report = {
            "protocol": protocol, "split_seed": seed, "fingerprint": fingerprint,
            "before": before_stats, "train_before_filter": train_before,
            "train_after_filter": train_after,
            "validation_raw": len(val_data or []), "test_raw": len(test_data),
            "validation_known": sum(u in known_users and i in known_items for u, i, _ in (val_data or [])),
            "test_known": sum(u in known_users and i in known_items for u, i, _ in test_data),
            "filter_scope": "training_only", "exclude_unknowns": True,
        }
    else:
        ratio_split = RatioSplit(data=filtered_feedback, test_size=test_size, val_size=val_size, **options)
        ratio_split.preprocessing_report = {"protocol": protocol, "before": before_stats,
                                             "after": feedback_statistics(filtered_feedback),
                                             "filter_scope": "before_split", "split_seed": seed}

    import hashlib
    ratio_split.preprocessing_report.update({
        "split_source": "provided_x_label" if provided_splits is not None else "random",
        "feedback": "binary" if protocol == "corrected" else "ratings",
        "source_metadata": source_metadata or {},
        "feature_matched_interactions": len(filtered_feedback),
        "source_interactions": len(feedback),
    })
    feature_digest = hashlib.sha256()
    for features in [mm_visuals, mm_text_embedings]:
        array = np.ascontiguousarray(features)
        feature_digest.update(str((array.shape, array.dtype)).encode())
        feature_digest.update(array.tobytes())
    ratio_split.preprocessing_report["features_sha256"] = feature_digest.hexdigest()
    if protocol == "legacy":
        import json
        ratio_split.preprocessing_report["fingerprint"] = hashlib.sha256(
            json.dumps(filtered_feedback, ensure_ascii=False).encode()).hexdigest()

    fallback_texts = raw_texts if raw_texts is not None else (texts if not texts_embeded else [f"Item {i}" for i in text_ids])
    item_idx_to_raw_text = _idx2raw_dict(fallback_texts, text_ids, ratio_split.train_set.iid_map)  # type: ignore
    ratio_split.train_set.item_idx_to_raw_text = item_idx_to_raw_text  # type: ignore

    return ratio_split, mm_visuals.mean(axis=0)  # type: ignore


def split_feedback(feedback, val_size=0.1, test_size=0.2, seed=42):
    """Random interaction split before any popularity-based filtering."""
    n = len(feedback)
    n_test, n_val = int(np.ceil(n * test_size)), int(np.ceil(n * val_size))
    if n_test < 1 or n - n_test - n_val < 1:
        raise ValueError("Not enough interactions for requested split")
    order = np.random.RandomState(seed).permutation(n)
    train = [feedback[i] for i in order[:n - n_test - n_val]]
    val = [feedback[i] for i in order[n - n_test - n_val:n - n_test]] if n_val else None
    test = [feedback[i] for i in order[n - n_test:]]
    return train, val, test


def feedback_statistics(feedback):
    counts = np.sort(np.asarray(list(Counter(i for _, i, _ in feedback).values()), dtype=float))
    users = len({u for u, _, _ in feedback})
    n = len(counts)
    gini = float((2 * np.dot(np.arange(1, n + 1), counts) / (n * counts.sum())) - (n + 1) / n) if n else 0.0
    return {"interactions": len(feedback), "users": users, "items": n, "gini": gini,
            "sparsity": 1 - len(feedback) / (users * n) if users and n else 1.0,
            "max_item_degree": int(counts[-1]) if n else 0}


def evaluate_recommendations(model, train_set, heldout_set, ks=(20, 50), exclude_set=None):
    """Full training-catalog ranking; groups use training degrees only.

    Head/middle/tail are the top 20%, next 30%, and remaining 50% of
    training items by degree (ties by item index). Group Recall/NDCG use
    the full recommendation list and average over users with group positives.
    exclude_set is validation history when evaluating the final test set.
    """
    import time
    started = time.perf_counter()
    if heldout_set is None:
        raise ValueError("A nonempty heldout set is required")
    ks = tuple(sorted(set(int(k) for k in ks)))
    n_items = train_set.num_items
    if not ks or ks[0] < 1:
        raise ValueError("Ranking cutoffs must be positive")
    train = train_set.matrix.tocsr()
    extra = exclude_set.matrix.tocsr() if exclude_set is not None else None
    degrees = np.asarray((train > 0).sum(axis=0)).ravel().astype(float)
    by_degree = np.lexsort((np.arange(n_items), -degrees))
    groups = np.full(n_items, 2, dtype=int)
    groups[by_degree[:int(np.ceil(n_items * 0.2))]] = 0
    groups[by_degree[int(np.ceil(n_items * 0.2)):int(np.ceil(n_items * 0.5))]] = 1
    positives = {}
    for u, i, r in zip(*heldout_set.uir_tuple):
        if r >= 1 and u < train_set.num_users and i < n_items:
            positives.setdefault(int(u), set()).add(int(i))
    totals, per_user = {}, {}
    exposures = {k: np.zeros(n_items, dtype=np.int64) for k in ks}
    for u, truth in sorted(positives.items()):
        scores = np.asarray(model.score(u), dtype=float)[:n_items].copy()
        if scores.shape != (n_items,) or not np.isfinite(scores).all():
            raise ValueError("Model returned invalid full-catalog scores")
        blocked = train.getrow(u).indices
        scores[blocked] = -np.inf
        if extra is not None and u < extra.shape[0]:
            blocked_extra = extra.getrow(u).indices
            scores[blocked_extra[blocked_extra < n_items]] = -np.inf
        truth = {i for i in truth if np.isfinite(scores[i])}
        if not truth:
            continue
        eligible = np.flatnonzero(np.isfinite(scores))
        size = min(ks[-1], len(eligible))
        if size:
            threshold = np.partition(scores[eligible], -size)[-size]
            above = eligible[scores[eligible] > threshold]
            tied = eligible[scores[eligible] == threshold]
            selected = np.concatenate([above, tied[:size - len(above)]])
        else:
            selected = eligible
        ranking = selected[np.lexsort((selected, -scores[selected]))]
        for k in ks:
            rec = ranking[:k]
            exposures[k][rec] += 1
            discounts = 1 / np.log2(np.arange(len(rec)) + 2)
            for group, label in [(None, ""), (0, "Head"), (1, "Middle"), (2, "Tail")]:
                target = truth if group is None else {i for i in truth if groups[i] == group}
                if not target:
                    continue
                hits = np.asarray([i in target for i in rec], dtype=float)
                ideal = (1 / np.log2(np.arange(min(k, len(target))) + 2)).sum()
                values = {f"{label}Recall@{k}": float(hits.sum() / len(target)),
                          f"{label}NDCG@{k}": float(np.dot(hits, discounts) / ideal)}
                for key, value in values.items():
                    per_user.setdefault(key, {})[str(u)] = value
    for key, values in per_user.items():
        totals[key] = float(np.mean(list(values.values())))
    for k, counts in exposures.items():
        total = counts.sum()
        totals[f"CatalogCoverage@{k}"] = float(np.count_nonzero(counts) / n_items)
        totals[f"ARP@{k}"] = float(np.dot(counts, degrees) / total) if total else 0.0
        totals[f"TailShare@{k}"] = float(counts[groups == 2].sum() / total) if total else 0.0
        for group, label in enumerate(["Head", "Middle", "Tail"]):
            mask = groups == group
            totals[f"{label}Coverage@{k}"] = float(np.count_nonzero(counts[mask]) / mask.sum()) if mask.any() else None
        ordered = np.sort(counts)
        totals[f"RecommendationGini@{k}"] = float(2 * np.dot(np.arange(1, n_items + 1), ordered) / (n_items * total) - (n_items + 1) / n_items) if total else 0.0
    if f"Recall@{ks[0]}" not in per_user:
        raise ValueError("Heldout split contains no evaluable users after exclusions")
    return {"metrics": totals, "per_user": per_user, "seconds": time.perf_counter() - started,
            "evaluated_users": len(per_user.get(f"Recall@{ks[0]}", {})),
            "candidate_protocol": "all_training_items_excluding_train_and_optional_validation",
            "groups": "training_degree_rank_20_30_50_percent"}
