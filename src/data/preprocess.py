import csv
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

IMAGES_DIR = PROJECT_ROOT / "data/raw/xview2/train/train/images"
LABELS_DIR = PROJECT_ROOT / "data/raw/xview2/train/train/labels"

OUTPUT_DIR = PROJECT_ROOT / "data/processed"
OUTPUT_FILE = OUTPUT_DIR / "manifest.csv"


DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_buildings(label_data):
    buildings = {}

    for feature in label_data.get("features", {}).get("xy", []):
        properties = feature.get("properties", {})

        if properties.get("feature_type") != "building":
            continue

        uid = properties.get("uid")

        if not uid:
            continue

        buildings[uid] = feature

    return buildings


def build_manifest():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pre_images = sorted(IMAGES_DIR.glob("*_pre_disaster.png"))

    rows = []

    skipped_unclassified = 0
    skipped_missing_post = 0

    for pre_image in pre_images:

        scene_id = pre_image.stem.replace("_pre_disaster", "")

        post_image = IMAGES_DIR / f"{scene_id}_post_disaster.png"

        pre_json = LABELS_DIR / f"{scene_id}_pre_disaster.json"
        post_json = LABELS_DIR / f"{scene_id}_post_disaster.json"

        if not post_image.exists():
            skipped_missing_post += 1
            continue

        if not pre_json.exists() or not post_json.exists():
            continue

        pre_data = load_json(pre_json)
        post_data = load_json(post_json)

        pre_buildings = get_buildings(pre_data)
        post_buildings = get_buildings(post_data)

        for uid, pre_feature in pre_buildings.items():

            post_feature = post_buildings.get(uid)

            if post_feature is None:
                continue

            post_properties = post_feature.get("properties", {})
            subtype = post_properties.get("subtype")

            if subtype not in DAMAGE_CLASSES:
                skipped_unclassified += 1
                continue

            rows.append({
                "scene_id": scene_id,
                "uid": uid,
                "pre_image": str(
                    pre_image.relative_to(PROJECT_ROOT)
                ),
                "post_image": str(
                    post_image.relative_to(PROJECT_ROOT)
                ),
                "pre_label": str(
                    pre_json.relative_to(PROJECT_ROOT)
                ),
                "post_label": str(
                    post_json.relative_to(PROJECT_ROOT)
                ),
                "damage_class": DAMAGE_CLASSES[subtype],
                "damage_label": subtype,
                "polygon_wkt": pre_feature.get("wkt", ""),
            })

    fieldnames = [
        "scene_id",
        "uid",
        "pre_image",
        "post_image",
        "pre_label",
        "post_label",
        "damage_class",
        "damage_label",
        "polygon_wkt",
    ]

    with open(OUTPUT_FILE, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Manifest created: {OUTPUT_FILE}")
    print(f"Total samples: {len(rows)}")
    print(f"Skipped unclassified: {skipped_unclassified}")
    print(f"Skipped missing post: {skipped_missing_post}")

    print("\nDamage distribution:")

    counts = {}

    for row in rows:
        label = row["damage_label"]
        counts[label] = counts.get(label, 0) + 1

    for label, count in counts.items():
        print(f"{label}: {count}")


if __name__ == "__main__":
    build_manifest()
