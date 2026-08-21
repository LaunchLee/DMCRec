import os, pickle, random, torch
import warnings
import numpy as np
import pandas as pd
import joblib, random
from collections import Counter
from cornac.data import TextModality, ImageModality
from cornac.eval_methods import RatioSplit


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


def encode_texts(texts, name, cache_dir="cache", model_name="intfloat/e5-base", batch_size=256, verbose=False):
    warnings.filterwarnings("ignore", module="transformers")
    warnings.filterwarnings("ignore", module="huggingface_hub")

    from sentence_transformers import SentenceTransformer

    embedding_dir = create_dir_if_not_exist(cache_dir, "_embeddings")
    save_path = os.path.join(embedding_dir, f"{name}_embeddings.joblib")
    if os.path.exists(save_path):
        if verbose:
            print(f"[INFO] Reusing text embeddings from {save_path}")
        return joblib.load(save_path)

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
    joblib.dump(embeddings, save_path)
    return embeddings


def multimodalize(texts, text_ids, visuals, visual_ids):
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

    return np.array(mm_ids), np.array(mm_texts), np.array(mm_visuals)


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


def dataset_preprocessing(name, visuals, visual_ids, texts, text_ids, feedback, *, texts_embeded=False, raw_texts=None, text_enc_model="intfloat/e5-base", cache_dir="cache", batch_size=256, max_samples=None, make_long_tail=True, k_core=None, seed=None, verbose=False):
    if verbose:
        print('-' * 79)

    text_embedings = texts if texts_embeded else encode_texts(texts, f"{name}_text", model_name=text_enc_model, cache_dir=cache_dir, batch_size=batch_size, verbose=verbose)

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
    filtered_feedback = [
        item for item in feedback if item[1] in valid_item_sets
    ]

    if make_long_tail:
        filtered_feedback = make_dataset_long_tail(
            feedback_data=filtered_feedback,
            seed=seed,
            verbose=verbose
        )

    if k_core is not None:
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

    ratio_split = RatioSplit(
        data=filtered_feedback,
        test_size=0.2,
        rating_threshold=1.0,
        item_text=text_modality,
        item_image=visual_modality,
        exclude_unknowns=True,
        verbose=verbose,
        seed=seed
    )

    fallback_texts = raw_texts if raw_texts is not None else (texts if not texts_embeded else [f"Item {i}" for i in text_ids])
    item_idx_to_raw_text = _idx2raw_dict(fallback_texts, text_ids, ratio_split.train_set.iid_map)  # type: ignore
    ratio_split.train_set.item_idx_to_raw_text = item_idx_to_raw_text  # type: ignore

    return ratio_split, mm_visuals.mean(axis=0)  # type: ignore
