import csv
import random
from collections import Counter, defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / "manifest.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

TRAIN_FILE = OUTPUT_DIR / "train.csv"
VAL_FILE = OUTPUT_DIR / "val.csv"

TRAIN_RATIO = 0.8
SEED = 42


def load_manifest():
    with open(MANIFEST_FILE, "r", newline="") as f:
        return list(csv.DictReader(f))


def split_scenes(rows):
    scenes = defaultdict(list)

    for row in rows:
        scenes[row["scene_id"]].append(row)

    scene_ids = list(scenes.keys())

    rng = random.Random(SEED)
    rng.shuffle(scene_ids)

    split_index = int(len(scene_ids) * TRAIN_RATIO)

    train_scenes = set(scene_ids[:split_index])
    val_scenes = set(scene_ids[split_index:])

    train_rows = []
    val_rows = []

    for scene_id, scene_rows in scenes.items():
        if scene_id in train_scenes:
            train_rows.extend(scene_rows)
        else:
            val_rows.extend(scene_rows)

    return train_rows, val_rows


def save_csv(rows, path):
    if not rows:
        raise ValueError(f"No rows to save: {path}")

    fieldnames = rows[0].keys()

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(name, rows):
    scenes = {row["scene_id"] for row in rows}
    classes = Counter(row["damage_label"] for row in rows)

    print(f"\n{name}")
    print("-" * 40)
    print(f"Scenes:    {len(scenes)}")
    print(f"Buildings: {len(rows)}")

    print("Damage distribution:")
    for label, count in sorted(classes.items()):
        percentage = count / len(rows) * 100
        print(f"  {label:<15} {count:>7} ({percentage:5.1f}%)")


def main():
    if not MANIFEST_FILE.exists():
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_FILE}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_manifest()

    print(f"Loaded samples: {len(rows)}")

    train_rows, val_rows = split_scenes(rows)

    train_scenes = {row["scene_id"] for row in train_rows}
    val_scenes = {row["scene_id"] for row in val_rows}

    overlap = train_scenes & val_scenes

    if overlap:
        raise RuntimeError(
            f"Scene leakage detected: {len(overlap)} scenes"
        )

    save_csv(train_rows, TRAIN_FILE)
    save_csv(val_rows, VAL_FILE)

    print_summary("TRAIN", train_rows)
    print_summary("VALIDATION", val_rows)

    print("\nLeakage check")
    print("-" * 40)
    print(f"Shared scenes: {len(overlap)}")

    print("\nFiles created:")
    print(f"  {TRAIN_FILE}")
    print(f"  {VAL_FILE}")


if __name__ == "__main__":
    main()