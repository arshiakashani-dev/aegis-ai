import csv
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"

CACHE_DIR = PROJECT_ROOT / "data" / "processed" / "crops"

IMAGE_SIZE = 128
PADDING = 16


def polygon_to_bbox(wkt):
    coordinate_text = wkt.replace("POLYGON ((", "").replace("))", "")

    points = []

    for point in coordinate_text.split(","):
        x, y = point.strip().split()
        points.append((float(x), float(y)))

    xs = [point[0] for point in points]
    ys = [point[1] for point in points]

    return min(xs), min(ys), max(xs), max(ys)


def crop_image(image, bbox):
    width, height = image.size

    x1, y1, x2, y2 = bbox

    x1 = max(0, int(x1 - PADDING))
    y1 = max(0, int(y1 - PADDING))
    x2 = min(width, int(x2 + PADDING))
    y2 = min(height, int(y2 + PADDING))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(
            f"Invalid crop: {(x1, y1, x2, y2)}"
        )

    crop = image.crop((x1, y1, x2, y2))

    return crop.resize(
        (IMAGE_SIZE, IMAGE_SIZE),
        Image.Resampling.BILINEAR,
    )


def process_split(csv_file, split_name):
    output_dir = CACHE_DIR / split_name
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(csv_file, "r", newline="") as f:
        rows = list(csv.DictReader(f))

    total = len(rows)

    print(f"\nProcessing {split_name}: {total} samples")

    current_scene = None
    pre_image = None
    post_image = None

    for index, row in enumerate(rows, start=1):
        scene_id = row["scene_id"]
        uid = row["uid"]

        if scene_id != current_scene:
            if pre_image is not None:
                pre_image.close()

            if post_image is not None:
                post_image.close()

            pre_path = PROJECT_ROOT / row["pre_image"]
            post_path = PROJECT_ROOT / row["post_image"]

            pre_image = Image.open(pre_path).convert("RGB")
            post_image = Image.open(post_path).convert("RGB")

            current_scene = scene_id

        bbox = polygon_to_bbox(row["polygon_wkt"])

        pre_crop = crop_image(pre_image, bbox)
        post_crop = crop_image(post_image, bbox)

        sample_dir = output_dir / scene_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        pre_output = sample_dir / f"{uid}_pre.jpg"
        post_output = sample_dir / f"{uid}_post.jpg"

        pre_crop.save(
            pre_output,
            format="JPEG",
            quality=90,
        )

        post_crop.save(
            post_output,
            format="JPEG",
            quality=90,
        )

        if index % 1000 == 0 or index == total:
            print(
                f"{split_name}: "
                f"{index}/{total} "
                f"({index / total * 100:.1f}%)"
            )

    if pre_image is not None:
        pre_image.close()

    if post_image is not None:
        post_image.close()


def main():
    CACHE_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    process_split(
        TRAIN_FILE,
        "train",
    )

    process_split(
        VAL_FILE,
        "val",
    )

    print("\nCrop cache complete.")
    print(f"Location: {CACHE_DIR}")


if __name__ == "__main__":
    main()