import os
import csv
import json
import math
from collections import Counter
import joblib
import torch
import numpy as np
from PIL import Image
from typing import Literal

from cornac.datasets import amazon_clothing
from experiments.dmcrec.exutils import dataset_preprocessing


def load_cornac_clothing(cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False, **split_options):
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
        verbose=verbose,
        **split_options
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
    if os.path.exists(cache_path):
        if verbose:
            print(f"[INFO] Reusing visual embeddings from: {cache_path}")
        visual_dict = joblib.load(cache_path)
        if not isinstance(visual_dict, dict) or not visual_dict:
            raise ValueError(f"Expected a nonempty item->visual-feature cache: {cache_path}")
        normalized = {}
        dimension = None
        for item_id, values in visual_dict.items():
            item_id = str(item_id)
            feature = np.asarray(values, dtype=np.float32)
            if feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
                raise ValueError(f"Invalid cached visual feature {item_id}: {cache_path}")
            if dimension is not None and feature.size != dimension:
                raise ValueError(f"Inconsistent visual feature dimensions: {cache_path}")
            dimension = feature.size
            if item_id in normalized:
                raise ValueError(f"Duplicate normalized visual ID {item_id}: {cache_path}")
            normalized[item_id] = feature
        if valid_item_ids is not None:
            normalized = {i: v for i, v in normalized.items() if i in valid_item_ids}
            missing = set(valid_item_ids) - set(normalized)
            if missing:
                raise ValueError(f"Visual cache is missing {len(missing)} interaction items: {cache_path}")
        if not normalized:
            raise ValueError(f"Visual cache IDs do not match interaction items: {cache_path}")
        if verbose:
            print(f"[INFO] Reused {len(visual_dict)} visual features.")
        return normalized

    valid_extensions = (".jpg", ".jpeg", ".png", ".webp")
    if not os.path.isdir(image_dir):
        raise FileNotFoundError(f"Missing covers {image_dir}; provide the existing visual cache at {cache_path}")
    all_img_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(valid_extensions))
    if not all_img_files:
        raise FileNotFoundError(f"Cover directory is empty: {image_dir}; provide visual cache {cache_path}")
    # Cache-only reading does not import/download a visual encoder.
    from transformers import CLIPProcessor, CLIPVisionModelWithProjection
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    if verbose:
        print(f"[INFO] Extracting visual features using {model_id} on {device}...")
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)

    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPVisionModelWithProjection.from_pretrained(model_id).to(torch_device).eval()

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

    if not visual_dict:
        raise ValueError(f"No matching readable covers in {image_dir}")
    if valid_item_ids is not None and set(valid_item_ids) - set(visual_dict):
        raise ValueError(f"Some interaction items have no readable covers in {image_dir}; incomplete cache was not saved")
    joblib.dump(visual_dict, cache_path)
    if verbose:
        print(f"[INFO] Visual embeddings saved to: {cache_path}")
    return visual_dict


def _read_id_mapping(path, index_column):
    """Read tab-delimited source mappings without numeric inference on raw IDs."""
    if not os.path.exists(path):
        return {}
    mapping = {}
    raw_ids = set()
    with open(path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        columns = reader.fieldnames or []
        if index_column not in columns or len(columns) != 2:
            raise ValueError(f"Invalid ID mapping header: {path}")
        raw_column = next(c for c in columns if c != index_column)
        for row in reader:
            try:
                index = int(row[index_column])
                raw = row[raw_column].strip()
            except (ValueError, TypeError, AttributeError) as exc:
                raise ValueError(f"Invalid ID mapping at {path}:{reader.line_num}") from exc
            if index < 0 or not raw or index in mapping or raw in raw_ids:
                raise ValueError(f"Duplicate/invalid ID at {path}:{reader.line_num}")
            mapping[index] = raw
            raw_ids.add(raw)
    if not mapping:
        raise ValueError(f"Empty ID mapping: {path}")
    return mapping


def _load_mmrec_dataset(dataset_dir, inter_name, max_inter=None, sort_by_time=True,
                        implicit=True, split_source="random", return_metadata=False):
    if split_source not in {"random", "provided"}:
        raise ValueError("split_source must be random or provided")
    if max_inter is not None and max_inter < 1:
        raise ValueError("max_inter must be positive")
    item_mapping = _read_id_mapping(os.path.join(dataset_dir, "i_id_mapping.csv"), "itemID")
    user_mapping = _read_id_mapping(os.path.join(dataset_dir, "u_id_mapping.csv"), "userID")
    # Read-only mappings avoid materializing a second copy of large feature files.
    image_feats = np.load(os.path.join(dataset_dir, "image_feat.npy"), mmap_mode="r", allow_pickle=False)
    text_feats = np.load(os.path.join(dataset_dir, "text_feat.npy"), mmap_mode="r", allow_pickle=False)
    if image_feats.ndim != 2 or text_feats.ndim != 2 or len(image_feats) != len(text_feats):
        raise ValueError(f"Image/text feature rows do not match: {dataset_dir}")
    num_items = len(image_feats)
    if item_mapping and set(item_mapping) != set(range(num_items)):
        raise ValueError(f"Item mapping does not match feature row indices: {dataset_dir}")
    raw_feedback = []
    label_counts = Counter()
    rating_counts = Counter()
    inter_path = os.path.join(dataset_dir, f"{inter_name}.inter")
    with open(inter_path, encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        fields = set(reader.fieldnames or [])
        if not {"userID", "itemID", "rating"} <= fields:
            raise ValueError(f"Expected userID/itemID/rating columns: {inter_path}")
        if split_source == "provided" and "x_label" not in fields:
            raise ValueError(f"Provided split requested but x_label is absent: {inter_path}")
        has_timestamp = "timestamp" in fields
        for row in reader:
            try:
                user, item = int(row["userID"]), int(row["itemID"])
                rating = float(row["rating"])
                timestamp = float(row["timestamp"]) if has_timestamp else 0.0
                label = int(row["x_label"]) if "x_label" in fields else None
            except (ValueError, TypeError) as exc:
                raise ValueError(f"Invalid interaction at {inter_path}:{reader.line_num}") from exc
            if user < 0 or not 0 <= item < num_items or not math.isfinite(rating) or not math.isfinite(timestamp):
                raise ValueError(f"Invalid ID/rating/timestamp at {inter_path}:{reader.line_num}")
            if user_mapping and user not in user_mapping:
                raise ValueError(f"Unmapped user {user}: {inter_path}:{reader.line_num}")
            if split_source == "provided" and label not in {0, 1, 2}:
                raise ValueError(f"Expected x_label 0/1/2: {inter_path}:{reader.line_num}")
            rating_counts[str(rating)] += 1
            label_counts[str(label)] += 1
            if implicit and rating < 1:
                continue
            raw_feedback.append((user_mapping.get(user, str(user)), item_mapping.get(item, str(item)),
                                 1.0 if implicit else rating, timestamp, label))
    if max_inter is not None and max_inter < len(raw_feedback):
        if sort_by_time and has_timestamp:
            raw_feedback.sort(key=lambda row: row[3], reverse=True)
        raw_feedback = raw_feedback[:max_inter]
    feedback = [(u, i, r) for u, i, r, _, _ in raw_feedback]
    metadata = {"split_source": split_source, "source_label_counts": dict(label_counts),
                "source_rating_counts": dict(rating_counts), "feedback": "binary" if implicit else "ratings",
                "item_identity": "source_mapping" if item_mapping else "local_integer_id",
                "user_identity": "source_mapping" if user_mapping else "local_integer_id"}
    if split_source == "provided":
        metadata["provided_splits"] = tuple(
            [feedback[n] for n, row in enumerate(raw_feedback) if row[4] == label] for label in (0, 1, 2))
        metadata["label_mapping"] = {"0": "train", "1": "validation", "2": "test"}
    visual_ids = [item_mapping.get(i, str(i)) for i in range(num_items)]
    result = (feedback, image_feats, visual_ids, text_feats, visual_ids.copy(),
              [f"Item ID: {vid}" for vid in visual_ids])
    return (*result, metadata) if return_metadata else result


def _iter_visual_json(path, chunk_size=65536):
    """Stream an item->vector JSON object without loading the whole text."""
    decoder = json.JSONDecoder()
    with open(path, encoding="utf-8-sig") as handle:
        buffer, position, eof = "", 0, False

        def fill():
            nonlocal buffer, position, eof
            chunk = handle.read(chunk_size)
            buffer = buffer[position:] + chunk
            position = 0
            eof = not chunk

        def whitespace():
            nonlocal position
            while True:
                while position < len(buffer) and buffer[position].isspace():
                    position += 1
                if position < len(buffer) or eof:
                    return
                fill()

        def consume(character):
            nonlocal position
            whitespace()
            if position >= len(buffer) or buffer[position] != character:
                raise ValueError(f"Expected {character!r} in {path}")
            position += 1

        def value():
            nonlocal position
            whitespace()
            while True:
                try:
                    result, end = decoder.raw_decode(buffer, position)
                    position = end
                    return result
                except json.JSONDecodeError as exc:
                    if eof:
                        raise ValueError(f"Invalid/truncated visual JSON: {path}") from exc
                    fill()

        consume("{")
        whitespace()
        if position < len(buffer) and buffer[position] == "}":
            position += 1
        else:
            while True:
                key = value()
                if not isinstance(key, str):
                    raise ValueError(f"Visual JSON keys must be item IDs: {path}")
                consume(":")
                yield key, value()
                whitespace()
                if position < len(buffer) and buffer[position] == "}":
                    position += 1
                    break
                consume(",")
        whitespace()
        if position != len(buffer):
            raise ValueError(f"Trailing data in visual JSON: {path}")


def load_pixel200k(pixel200k_dir="datasets/Pixel200k", cache_dir="cache",  max_inter = None, sort_by_time: bool = True, batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False, **split_options):
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
    visual_keys = set()
    feature_dim = None
    for item_id, values in _iter_visual_json(PIXEL200K_VISUAL):
        item_id = item_id.strip()
        feature = np.asarray(values, dtype=np.float32)
        if not item_id or item_id in visual_keys or feature.ndim != 1 or not feature.size or not np.isfinite(feature).all():
            raise ValueError(f"Invalid/duplicate Pixel visual feature: {item_id}")
        if feature_dim is not None and feature.size != feature_dim:
            raise ValueError(f"Inconsistent Pixel visual dimensions: {item_id}")
        feature_dim = feature.size
        visual_keys.add(item_id)
        pixel_visual_ids.append(item_id)
        pixel_visuals.append(feature)

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
        verbose=verbose,
        **split_options
    )

    return ratio_split, mm_visuals_mean


def load_microlens50k(mlens50k_dir="datasets/MicroLens-50k", cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False, **split_options):
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
        verbose=verbose,
        **split_options
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
                if not row:
                    continue
                if len(row) != 3 or not row[0].strip() or not row[1].strip():
                    raise ValueError(f"Invalid item,user,timestamp row: {pair_file}:{reader.line_num}")
                if not math.isfinite(float(row[2])):
                    raise ValueError(f"Invalid timestamp: {pair_file}:{reader.line_num}")
                i_id = f"{cat}_i_{row[0].strip()}"
                u_id = f"{cat}_u_{row[1].strip()}"
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

        missing_metadata = active_items - set(text_dict)
        if missing_metadata:
            raise ValueError(f"{prefix}: {len(missing_metadata)} interaction items have no metadata; check pair column order")

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
            raise ValueError(f"No usable visual features for {prefix}; domain cannot be silently skipped")

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


def load_ninerec_bili(ninerec_dir="datasets/NineRec-Downstream-Bili", cache_dir="cache", batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False, **split_options):
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
        verbose=verbose,
        **split_options
    )

    return ratio_split, mm_visuals_mean


def load_mmrec(datasets_dir="datasets", which: Literal['Baby', 'Sports', 'Clothing', 'Elec']="Baby", cache_dir="cache", max_inter = None, sort_by_time: bool = True, batch_size=256, max_samples=None, k_core=None, make_long_tail=True, seed=None, verbose=False, split_source="random", **split_options):
    if split_source == "provided" and split_options.get("protocol", "corrected") == "legacy":
        raise ValueError("Provided x_label splits require the corrected protocol")
    dataset_dir = os.path.join(datasets_dir, f"MMRec-{which}")
    (mmrec_feedback,
     mmrec_visuals, mmrec_visual_ids,
     mmrec_texts,   mmrec_text_ids,
     mmrec_dummy_raw_texts, source_metadata) = _load_mmrec_dataset(
        dataset_dir,
        which.lower(),
        max_inter=max_inter,
        sort_by_time=sort_by_time,
        implicit=split_options.get("protocol", "corrected") != "legacy",
        split_source=split_source,
        return_metadata=True,
    )
    provided_splits = source_metadata.pop("provided_splits", None)

    ratio_split, mm_visuals_mean = dataset_preprocessing(
        f"mmrec_{which.lower()}",
        mmrec_visuals,
        mmrec_visual_ids,
        mmrec_texts,
        mmrec_text_ids,
        mmrec_feedback,
        texts_embeded=True,
        raw_texts=mmrec_dummy_raw_texts,
        provided_splits=provided_splits,
        source_metadata=source_metadata,
        batch_size=batch_size,
        cache_dir=cache_dir,
        max_samples=max_samples,
        k_core=k_core,
        make_long_tail=make_long_tail,
        seed=seed,
        verbose=verbose,
        **split_options
    )

    return ratio_split, mm_visuals_mean
