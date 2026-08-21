import os
import csv
import json
import joblib
import torch
import numpy as np
import pandas as pd
from PIL import Image
from transformers import CLIPProcessor, CLIPVisionModelWithProjection
from typing import Literal

from cornac.datasets import amazon_clothing
from misc import dataset_preprocessing


def load_cornac_clothing(cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False):
    clothing_feedback = amazon_clothing.load_feedback()  # list[(user_id, item_id, rating)]
    clothing_texts, clothing_text_ids = amazon_clothing.load_text()  # (list[text], list[item_id])
    clothing_visuals, clothing_visual_ids = amazon_clothing.load_visual_feature()  # (list[image_feature], list[item_id])

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        "clothing",
        clothing_visuals,
        clothing_visual_ids,
        clothing_texts,
        clothing_text_ids,
        clothing_feedback,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        seed=seed,
        verbose=verbose
    )
    return ratio_split, mm_visuals_mean


def _encode_image_folder(
    image_dir: str,
    cache_path: str,
    valid_item_ids: set = None,  # type: ignore
    model_id: str = "openai/clip-vit-base-patch32",
    batch_size: int = 64,
    device: str = None,  # type: ignore
    id_extractor = None,
    verbose = False
):
    """
    Args:
        image_dir (str): Image dir.
        cache_path (str): Save path (.joblib).
        valid_item_ids (set, optional): IDs that will be embeded.
        model_id (str): Model name on HuggingFace.
        batch_size (int): Batch size.
        device (str): devce (cuda/cpu). Auto select if None.
        id_extractor (callable, optional): A function that get item_id from the filename. Default item_id is the filename without ext.

    Returns:
        dict: {item_id: feature_vector}
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if os.path.exists(cache_path):
        if verbose:
            print(f"[INFO] Reusing visual embeddings from: {cache_path}")
        visual_dict = joblib.load(cache_path)
        if verbose:
            print(f"[INFO] Reused {len(visual_dict)} visual features.")
        return visual_dict

    if verbose:
        print(f"[INFO] Extracting visual features using {model_id} on {device}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    try:
        processor = CLIPProcessor.from_pretrained(model_id)
        model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(torch_device).eval()  # type: ignore
    except Exception as e:
        print(f"[ERROR] CLIP model init failed: {e}")
        return None

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    all_img_files = [f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions)]

    if not all_img_files:
        if verbose:
            print(f"[Warning] No valid images found in {image_dir}")
        return {}

    if verbose:
        print(f"[INFO] Found {len(all_img_files)} images. Starting batch encoding...")

    if id_extractor is None:
        id_extractor = lambda filename: os.path.splitext(filename)[0].strip()

    visual_dict = {}

    from tqdm.auto import tqdm
    with torch.no_grad(), tqdm(total=len(all_img_files), desc="CLIP Encoding", unit="img") as pbar:
        for i in range(0, len(all_img_files), batch_size):
            batch_files = all_img_files[i : i + batch_size]
            batch_images = []
            batch_item_ids = []

            for file_name in batch_files:
                item_id = id_extractor(file_name)

                if valid_item_ids is not None and item_id not in valid_item_ids:
                    continue

                img_path = os.path.join(image_dir, file_name)
                try:
                    img = Image.open(img_path).convert("RGB")
                    batch_images.append(img)
                    batch_item_ids.append(item_id)
                except Exception as e:
                    print(f"\n[Warning] Failed to load {file_name}: {e}")
                    continue

            if batch_images:
                inputs = processor(images=batch_images, return_tensors="pt").to(torch_device)  # type: ignore
                outputs = model(**inputs)
                image_features = outputs.image_embeds.cpu().numpy()  # [Batch, Embed_Dim]

                for item_id, feat_vector in zip(batch_item_ids, image_features):
                    visual_dict[item_id] = feat_vector

            pbar.update(len(batch_files))
            pbar.set_postfix({"Processed": len(visual_dict)})

    joblib.dump(visual_dict, cache_path)
    if verbose:
        print(f"[INFO] Visual embeddings saved to: {cache_path}")
    return visual_dict


def _load_mmrec_dataset(dataset_dir: str, inter_name: str, max_inter = None, sort_by_time: bool = True):
    i_map_path = os.path.join(dataset_dir, 'i_id_mapping.csv')
    u_map_path = os.path.join(dataset_dir, 'u_id_mapping.csv')
    inter_path = os.path.join(dataset_dir, f'{inter_name}.inter')

    item_id_to_asin = {}
    if os.path.exists(i_map_path):
        try:
            df_i = pd.read_csv(i_map_path, sep=None, engine='python')
            cols = df_i.columns
            for _, row in df_i.iterrows():
                raw_id = str(row[cols[0]]).strip()
                int_id = int(row[cols[1]])
                item_id_to_asin[int_id] = raw_id
        except Exception as e:
            print(f"[Warning] Failed to parse {i_map_path}: {e}")

    user_id_to_raw = {}
    if os.path.exists(u_map_path):
        try:
            df_u = pd.read_csv(u_map_path, sep=None, engine='python')
            cols = df_u.columns
            for _, row in df_u.iterrows():
                raw_id = str(row[cols[0]]).strip()
                int_id = int(row[cols[1]])
                user_id_to_raw[int_id] = raw_id
        except Exception as e:
            print(f"[Warning] Failed to parse {u_map_path}: {e}")

    raw_feedback_list = []
    has_timestamp = False

    with open(inter_path, 'r', encoding='utf-8') as f:
        header = next(f, None)
        for line in f:
            line_str = line.strip()
            if not line_str:
                continue

            parts = line_str.replace('\t', ' ').split()
            if len(parts) >= 3:
                try:
                    u_id = int(parts[0])
                    i_id = int(parts[1])
                    rating = float(parts[2])

                    ts = float(parts[3]) if len(parts) >= 4 else 0.0
                    if len(parts) >= 4:
                        has_timestamp = True

                    raw_u = user_id_to_raw.get(u_id, str(u_id))
                    raw_i = item_id_to_asin.get(i_id, str(i_id))

                    raw_feedback_list.append((raw_u, raw_i, rating, ts))
                except ValueError:
                    continue

    if max_inter is not None and max_inter < len(raw_feedback_list):
        if sort_by_time and has_timestamp:
            raw_feedback_list.sort(key=lambda x: x[3], reverse=True)
            raw_feedback_list = raw_feedback_list[:max_inter]
        else:
            raw_feedback_list = raw_feedback_list[:max_inter]

    feedback = [(u, i, r) for u, i, r, _ in raw_feedback_list]

    v_path = os.path.join(dataset_dir, 'image_feat.npy')
    t_path = os.path.join(dataset_dir, 'text_feat.npy')

    if not os.path.exists(v_path) or not os.path.exists(t_path):
        raise FileNotFoundError(f"Missing image_feat.npy or text_feat.npy in {dataset_dir}")

    image_feats = np.load(v_path)
    text_feats = np.load(t_path)

    num_items = image_feats.shape[0]
    visual_ids = [item_id_to_asin.get(i, str(i)) for i in range(num_items)]
    text_ids = visual_ids.copy()

    dummy_raw_texts = [f"Item ID: {vid}" for vid in visual_ids]

    return feedback, image_feats, visual_ids, text_feats, text_ids, dummy_raw_texts


def load_pixel200k(pixel200k_dir="datasets/Pixel200k", cache_dir="cache",  max_inter = None, sort_by_time: bool = True, batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False):
    PIXEL200K_INTER = f"{pixel200k_dir}/Pixel200k_inter.csv"  # item_id, user_id, timestamp
    PIXEL200K_TEXT = f"{pixel200k_dir}/Pixel200k_text.csv"  # item_id, view_number, comment_number, thumbup_number, share_number, coin_number, favorite_number, barrage_number, title, tag, description
    PIXEL200K_VISUAL = f"{pixel200k_dir}/pixel200k_image_feature.json"  # Encoded by CLIP-RN50 according to the author, with 1024 dims

    pixel_feedback = []
    with open(PIXEL200K_INTER, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if max_inter is not None and sort_by_time:
            inter_list = []
            for row in reader:
                user_id = row["user_id"].strip()
                item_id = row["item_id"].strip()
                ts = float(row.get("timestamp", 0))
                inter_list.append((user_id, item_id, 1.0, ts))
            inter_list.sort(key=lambda x: x[3], reverse=True)
            inter_list = inter_list[:max_inter]
            pixel_feedback = [(u, i, r) for u, i, r, _ in inter_list]
        else:
            for row in reader:
                user_id = row["user_id"].strip()
                item_id = row["item_id"].strip()
                pixel_feedback.append((user_id, item_id, 1.0))
                if max_inter is not None and len(pixel_feedback) >= max_inter:
                    break
    if verbose:
        print("[INFO] Feedback count:", len(pixel_feedback))

    pixel_texts = []
    pixel_text_ids = []
    with open(PIXEL200K_TEXT, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item_id"].strip()
            title = row.get("title", "").strip()
            tag = row.get("tag", "").strip()
            desc = row.get("description", "").strip()

            combined_text = f"Title: {title}. Tags: {tag}. Description: {desc}."

            pixel_texts.append(combined_text)
            pixel_text_ids.append(item_id)
    if verbose:
        print("[INFO] Text count:", len(pixel_texts))

    pixel_visuals = []
    pixel_visual_ids = []
    with open(PIXEL200K_VISUAL, mode="r", encoding="utf-8") as f:
        visual_dict = json.load(f)

        for item_id, feat_array in visual_dict.items():
            pixel_visuals.append(np.array(feat_array, dtype=np.float32))
            pixel_visual_ids.append(item_id.strip())

    pixel_visuals = np.array(pixel_visuals, dtype=np.float32)
    if verbose:
        print("[INFO] Visual count:", len(pixel_visuals))

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        "pixel200k",
        pixel_visuals,
        pixel_visual_ids,
        pixel_texts,
        pixel_text_ids,
        pixel_feedback,
        batch_size=batch_size,
        cache_dir=cache_dir,
        seed=seed,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        verbose=verbose
    )

    return ratio_split, mm_visuals_mean


def load_microlens50k(mlens50k_dir="datasets/MicroLens-50k", cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False):
    mlens_feedback = []
    active_item_ids = set()

    MLENS50K_INTER = f"{mlens50k_dir}/MicroLens-50k_pairs.csv"  # user, item, timestamp
    MLENS50K_TEXT = f"{mlens50k_dir}/MicroLens-50k_titles.csv"  # item, title
    MLENS50K_IMAGES = f"{mlens50k_dir}/MicroLens-50k_covers"  # The folder contains covers. Filename is the item id that is just number starting from 1 for this dataset.
    MLENS50K_VISUAL_EMBEDDING_FILE = f"{cache_dir}/_embeddings/microlens50k_visual_embeddings.joblib"

    with open(MLENS50K_INTER, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            user_id = row["user"].strip()
            item_id = row["item"].strip()
            active_item_ids.add(item_id)
            mlens_feedback.append((user_id, item_id, 1.0))
    if verbose:
        print(f"[INFO] Feedback count: {len(mlens_feedback)}")


    visual_dict = _encode_image_folder(
        image_dir=MLENS50K_IMAGES,
        cache_path=MLENS50K_VISUAL_EMBEDDING_FILE,
        valid_item_ids=active_item_ids,
        model_id="openai/clip-vit-base-patch32",
        batch_size=batch_size,
        verbose=verbose
    )

    mlens_visuals = []
    mlens_visual_ids = []

    for item_id, feat_array in visual_dict.items():  # type: ignore
        mlens_visuals.append(feat_array)
        mlens_visual_ids.append(item_id)

    mlens_visuals = np.array(mlens_visuals, dtype=np.float32)
    if verbose:
        print(f"[INFO] Visual shape: {mlens_visuals.shape}")

    mlens_texts = []
    mlens_text_ids = []

    text_dict = {}
    with open(MLENS50K_TEXT, mode="r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            item_id = row["item"].strip()
            title = row["title"].strip()
            text_dict[item_id] = f"Title: {title}."

    for item_id in mlens_visual_ids:
        combined_text = text_dict.get(item_id, "Title: Unknown.")
        mlens_texts.append(combined_text)
        mlens_text_ids.append(item_id)
    if verbose:
        print(f"Text count: {len(mlens_text_ids)}")

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        "microlens50k",
        mlens_visuals,
        mlens_visual_ids,
        mlens_texts,
        mlens_text_ids,
        mlens_feedback,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        seed=seed,
        verbose=verbose
    )

    return ratio_split, mm_visuals_mean


def _load_unified_bili_data(base_dir, cache_dir, batch_size=256, verbose=False):
    CATEGORIES = ["Cartoon", "Dance", "Food", "Movie", "Music"]
    unified_feedback = []

    unified_visuals = []
    unified_visual_ids = []
    unified_texts = []
    unified_text_ids = []

    total_feedback_count = 0

    for cat in CATEGORIES:
        prefix = f"Bili_{cat}"
        cat_dir = os.path.join(base_dir, prefix)

        pair_file = os.path.join(cat_dir, f"{prefix}_pair.csv")
        item_file = os.path.join(cat_dir, f"{prefix}_item.csv")
        img_dir = os.path.join(cat_dir, f"{prefix}_cover")
        cache_file = os.path.join(cache_dir, f"_embeddings", f"ninerec_downstream_bili_{cat}_visual_embeddings.joblib")

        if verbose:
            print(f"\n{'='*79}\n[INFO] Processing Sub-dataset: {cat}\n{'='*79}")

        active_items = set()
        with open(pair_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                u_id = f"{cat}_u_{row[0].strip()}"
                i_id = f"{cat}_i_{row[1].strip()}"
                unified_feedback.append((u_id, i_id, 1.0))
                active_items.add(i_id)
                total_feedback_count += 1

        text_dict = {}
        with open(item_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) < 3: continue
                i_id = f"{cat}_i_{row[0].strip()}"
                if i_id not in active_items:
                    continue

                zh_title = row[1].strip()
                en_title = row[2].strip()
                text_dict[i_id] = f"Title: {zh_title}. English translation: {en_title}"

        def bili_id_extractor(filename, category=cat):
            raw_id = os.path.splitext(filename)[0].strip()
            return f"{category}_i_{raw_id}"

        visual_dict = _encode_image_folder(
            image_dir=img_dir,
            cache_path=cache_file,
            valid_item_ids=active_items,
            model_id="openai/clip-vit-base-patch32",
            batch_size=batch_size,
            id_extractor=bili_id_extractor,
            verbose=verbose
        )

        if not visual_dict:
            continue

        for i_id, v_feat in visual_dict.items():
            unified_visuals.append(v_feat)
            unified_visual_ids.append(i_id)

            t_feat = text_dict.get(i_id, "Title: Unknown")
            unified_texts.append(t_feat)
            unified_text_ids.append(i_id)

    unified_visuals = np.array(unified_visuals, dtype=np.float32)

    if verbose:
        print(f"\n[INFO] Unified Dataset Build Complete!")
        print(f"[INFO] Total Feedback: {total_feedback_count}")
        print(f"[INFO] Total Items (with aligned modalities): {len(unified_visual_ids)}")
        print(f"[INFO] Visual Shape: {unified_visuals.shape}")

    return unified_feedback, (unified_texts, unified_text_ids), (unified_visuals, unified_visual_ids)


def load_ninerec_bili(ninerec_dir="datasets/NineRec-Downstream-Bili", cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False):
    bili_feedback, (bili_texts, bili_text_ids), (bili_visuals, bili_visual_ids) = (
        _load_unified_bili_data(ninerec_dir, cache_dir, batch_size=batch_size, verbose=verbose)
    )

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        name="ninerec_downstream_bili",
        visuals=bili_visuals,
        visual_ids=bili_visual_ids,
        texts=bili_texts,
        text_ids=bili_text_ids,
        feedback=bili_feedback,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        seed=seed,
        verbose=verbose
    )

    return ratio_split, mm_visuals_mean


def load_mmrec(datasets_dir="datasets", which: Literal['Baby', 'Sports', 'Clothing', 'Elec']="Baby", cache_dir="cache", max_inter = None, sort_by_time: bool = True, batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False):
    dataset_dir = os.path.join(datasets_dir, f"MMRec-{which}")
    (mmrec_feedback,
     mmrec_visuals, mmrec_visual_ids,
     mmrec_texts,   mmrec_text_ids,
     mmrec_dummy_raw_texts) = _load_mmrec_dataset(
        dataset_dir,
        which.lower(),
        max_inter=max_inter,
        sort_by_time=sort_by_time
    )

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        "mmrec_sports",
        mmrec_visuals,
        mmrec_visual_ids,
        mmrec_texts,
        mmrec_text_ids,
        mmrec_feedback,
        texts_embeded=True,
        raw_texts=mmrec_dummy_raw_texts,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        seed=seed,
        verbose=verbose
    )

    return ratio_split, mm_visuals_mean
