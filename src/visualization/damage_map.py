import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENE = "palu-tsunami_00000087"
SPLIT = "val"

MANIFEST_FILE = PROJECT_ROOT / "data" / "processed" / f"{SPLIT}.csv"
RESULT_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "scene_inference"
    / f"{SCENE}.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "damage_maps"
OUTPUT_FILE = OUTPUT_DIR / f"{SCENE}_damage_map.png"

CLASS_COLORS = {
    "no-damage": (70, 200, 100),
    "minor-damage": (255, 220, 70),
    "major-damage": (255, 150, 40),
    "destroyed": (230, 60, 60),
}

CLASS_ORDER = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


def load_manifest():
    with open(MANIFEST_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        row["uid"]: row
        for row in rows
        if row["scene_id"] == SCENE
    }


def load_predictions():
    with open(RESULT_FILE) as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data
    elif isinstance(data, dict):
        records = (
            data.get("predictions")
            or data.get("results")
            or data.get("records")
            or []
        )
    else:
        raise ValueError("Unsupported inference JSON format")

    return {
        record["uid"]: record
        for record in records
        if "uid" in record
    }


def parse_polygon_wkt(wkt):
    """
    Parse xView2 POLYGON WKT into pixel coordinate pairs.

    Example:
    POLYGON ((x y, x y, ...))
    """
    if not wkt:
        return []

    match = re.search(r"POLYGON\s*\(\((.*?)\)\)", wkt)

    if not match:
        return []

    points = []

    for pair in match.group(1).split(","):
        parts = pair.strip().split()

        if len(parts) < 2:
            continue

        x = float(parts[0])
        y = float(parts[1])

        points.append((x, y))

    return points


def draw_legend(draw, image_width):
    x = 30
    y = 30

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            22,
        )
    except OSError:
        font = None

    title = "AEGIS AI — BUILDING DAMAGE MAP"
    draw.text(
        (x, y),
        title,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    y += 40

    for class_name in CLASS_ORDER:
        color = CLASS_COLORS[class_name]

        draw.rectangle(
            (x, y, x + 24, y + 24),
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )

        draw.text(
            (x + 35, y),
            class_name,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

        y += 32


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    manifest = load_manifest()
    predictions = load_predictions()

    if not manifest:
        raise RuntimeError(f"No manifest rows found for scene: {SCENE}")

    if not predictions:
        raise RuntimeError("No predictions found in inference JSON")

    first_row = next(iter(manifest.values()))

    post_image_path = PROJECT_ROOT / first_row["post_image"]

    image = Image.open(post_image_path).convert("RGB")
    overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    counts = {name: 0 for name in CLASS_ORDER}

    drawn = 0
    skipped = 0

    for uid, prediction in predictions.items():
        row = manifest.get(uid)

        if row is None:
            skipped += 1
            continue

        predicted_class = prediction.get("prediction")

        if predicted_class not in CLASS_COLORS:
            skipped += 1
            continue

        polygon = parse_polygon_wkt(row["polygon_wkt"])

        if len(polygon) < 3:
            skipped += 1
            continue

        color = CLASS_COLORS[predicted_class]

        # Transparent fill + solid outline.
        fill_color = (*color, 75)
        outline_color = (*color, 230)

        draw.polygon(
            polygon,
            fill=fill_color,
            outline=outline_color,
        )

        counts[predicted_class] += 1
        drawn += 1

    result = Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB")

    legend = ImageDraw.Draw(result)
    draw_legend(legend, result.width)

    result.save(OUTPUT_FILE, quality=95)

    print("Aegis AI - Damage Map")
    print("=" * 50)
    print(f"Scene:       {SCENE}")
    print(f"Buildings:   {len(predictions)}")
    print(f"Mapped:      {drawn}")
    print(f"Skipped:     {skipped}")
    print()
    print("Predicted Damage")
    print("-" * 50)

    for class_name in CLASS_ORDER:
        print(f"{class_name:<15}: {counts[class_name]}")

    print()
    print(f"Map saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
