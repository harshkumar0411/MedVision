import os
import shutil
import hashlib
import random
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

# ======================================
# CONFIGURATION
# Update SOURCE_ROOT if your renamed folders sit inside another directory
# ======================================
SOURCE_ROOT = r"C:\Users\LOQ\Data\Medical Imagining (CT scan, MRI, X-ray, and Microscopic Imagery) Data\Medical Imagining"
DEST_ROOT = "datasets"
SEED = 42
TRAIN_SPLIT_RATIO = 0.8  # Used whenever a class folder has no existing Train/Test split

SPECIALTIES = [
    "bone_break",
    "lung_cancer",
    "renal_malignancy",
    "skin_lesions",
]

TRAIN_ALIASES = {"train", "training"}
TEST_ALIASES = {"test", "testing", "val", "valid", "validation"}
VALID_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")

random.seed(SEED)


def verify_and_hash(filepath):
    """Validates image readability, converts to 3-channel RGB, and returns MD5 pixel hash."""
    try:
        with Image.open(filepath) as img:
            img.verify()
        with Image.open(filepath) as img:
            rgb_img = img.convert("RGB")
            if rgb_img.width < 32 or rgb_img.height < 32:
                return None, None
            pixel_hash = hashlib.md5(rgb_img.tobytes()).hexdigest()
            return pixel_hash, rgb_img
    except Exception:
        return None, None


def collect_images(directory):
    """Recursively collects all valid image files under a directory."""
    files_found = []
    for root, _, files in os.walk(directory):
        for fname in sorted(files):
            if fname.lower().endswith(VALID_EXTS):
                files_found.append(os.path.join(root, fname))
    return files_found


def write_clean_images(file_paths, out_dir, seen_hashes):
    """Deduplicates, standardizes to RGB JPEG, and writes to out_dir."""
    os.makedirs(out_dir, exist_ok=True)
    saved, corrupt, duplicates = 0, 0, 0

    for src_path in file_paths:
        img_hash, rgb_img = verify_and_hash(src_path)
        if img_hash is None:
            corrupt += 1
            continue
        if img_hash in seen_hashes:
            duplicates += 1
            continue

        seen_hashes.add(img_hash)
        out_path = os.path.join(out_dir, f"img_{saved:05d}.jpg")
        rgb_img.save(out_path, format="JPEG", quality=95)
        saved += 1

    return saved, corrupt, duplicates


def get_subfolders(path):
    """Returns sorted immediate subdirectories of path."""
    if not os.path.exists(path):
        return []
    return sorted([
        d for d in os.listdir(path)
        if os.path.isdir(os.path.join(path, d))
    ])


def prepare_specialty(specialty_name):
    specialty_dir = os.path.join(SOURCE_ROOT, specialty_name)
    if not os.path.exists(specialty_dir):
        print(f"[SKIP] Folder not found: {specialty_dir}")
        return

    print(f"\n=======================================================")
    print(f"PREPARING: {specialty_name}  -->  datasets/{specialty_name}")
    print(f"=======================================================")

    # Build into a staging directory first so SOURCE_ROOT == DEST_ROOT is 100% safe
    staging_dir = os.path.join(DEST_ROOT, f"_staging_{specialty_name}")
    if os.path.exists(staging_dir):
        shutil.rmtree(staging_dir)

    seen_hashes = set()
    total_corrupt = 0
    total_dups = 0

    top_subs = get_subfolders(specialty_dir)
    top_train_dirs = [d for d in top_subs if d.strip().lower() in TRAIN_ALIASES]
    top_test_dirs = [d for d in top_subs if d.strip().lower() in TEST_ALIASES]

    # ------------------------------------------------------------------
    # LAYOUT 1: Top-level folders are Train / Test (case-insensitive)
    # ------------------------------------------------------------------
    if top_train_dirs:
        train_root = os.path.join(specialty_dir, top_train_dirs[0])
        test_root = os.path.join(specialty_dir, top_test_dirs[0]) if top_test_dirs else None

        class_names = get_subfolders(train_root)
        for cls_name in class_names:
            clean_cls = cls_name.strip()
            train_files = collect_images(os.path.join(train_root, cls_name))
            test_files = (
                collect_images(os.path.join(test_root, cls_name))
                if test_root and os.path.exists(os.path.join(test_root, cls_name))
                else []
            )

            if not test_files:
                random.shuffle(train_files)
                cut = int(len(train_files) * TRAIN_SPLIT_RATIO)
                test_files = train_files[cut:]
                train_files = train_files[:cut]

            tr_ok, tr_bad, tr_dup = write_clean_images(
                train_files, os.path.join(staging_dir, "train", clean_cls), seen_hashes
            )
            te_ok, te_bad, te_dup = write_clean_images(
                test_files, os.path.join(staging_dir, "test", clean_cls), seen_hashes
            )
            total_corrupt += tr_bad + te_bad
            total_dups += tr_dup + te_dup
            print(f"  [Top-Split]  {clean_cls:<30} -> {tr_ok:4d} train | {te_ok:4d} test")

    # ------------------------------------------------------------------
    # LAYOUT 2 & 3: Top-level folders are Disease Classes
    # Each class can either contain Train/Test subfolders OR raw images
    # ------------------------------------------------------------------
    else:
        for cls_name in top_subs:
            clean_cls = cls_name.strip()
            cls_path = os.path.join(specialty_dir, cls_name)
            cls_subs = get_subfolders(cls_path)

            cls_train_dirs = [d for d in cls_subs if d.strip().lower() in TRAIN_ALIASES]
            cls_test_dirs = [d for d in cls_subs if d.strip().lower() in TEST_ALIASES]

            if cls_train_dirs:
                train_files = []
                for d in cls_train_dirs:
                    train_files.extend(collect_images(os.path.join(cls_path, d)))

                test_files = []
                for d in cls_test_dirs:
                    test_files.extend(collect_images(os.path.join(cls_path, d)))

                if not test_files:
                    random.shuffle(train_files)
                    cut = int(len(train_files) * TRAIN_SPLIT_RATIO)
                    test_files = train_files[cut:]
                    train_files = train_files[:cut]
                mode_tag = "[Sub-Split]"
            else:
                all_files = collect_images(cls_path)
                random.shuffle(all_files)
                cut = int(len(all_files) * TRAIN_SPLIT_RATIO)
                train_files = all_files[:cut]
                test_files = all_files[cut:]
                mode_tag = "[Auto-80/20]"

            tr_ok, tr_bad, tr_dup = write_clean_images(
                train_files, os.path.join(staging_dir, "train", clean_cls), seen_hashes
            )
            te_ok, te_bad, te_dup = write_clean_images(
                test_files, os.path.join(staging_dir, "test", clean_cls), seen_hashes
            )
            total_corrupt += tr_bad + te_bad
            total_dups += tr_dup + te_dup
            print(f"  {mode_tag} {clean_cls:<30} -> {tr_ok:4d} train | {te_ok:4d} test")

    # Replace final target directory atomically from staging
    final_dir = os.path.join(DEST_ROOT, specialty_name)
    if os.path.exists(final_dir):
        shutil.rmtree(final_dir)
    shutil.move(staging_dir, final_dir)

    print(f"  Summary: {total_dups} duplicates/leaks removed | {total_corrupt} corrupt files skipped")


if __name__ == "__main__":
    os.makedirs(DEST_ROOT, exist_ok=True)
    for specialty in SPECIALTIES:
        prepare_specialty(specialty)
    print("\nALL DATASETS PREPARED SUCCESSFULLY! Next step: python training/train.py")