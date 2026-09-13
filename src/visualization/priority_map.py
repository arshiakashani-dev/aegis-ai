import csv
import json
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

SCENE = "palu-tsunami_00000087"
SPLIT = "val"

MANIFEST_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / f"{SPLIT}.csv"
)

RISK_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "risk"
    / f"{SCENE}.json"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "priority_maps"

OUTPUT_FILE = (
    OUTPUT_DIR
    / f"{SCENE}_priority_map.png"
)


PRIORITY_COLORS = {
    "low": (70, 200, 100),
    "medium": (255, 220, 70),
    "high": (255, 150, 40),
    "critical": (230, 60, 60),
}

PRIORITY_ORDER = [
    "low",
    "medium",
    "high",
    "critical",
]


def load_manifest():
    with open(MANIFEST_FILE, newline="") as f:
        rows = list(csv.DictReader(f))

    return {
        row["uid"]: row
        for row in rows
        if row["scene_id"] == SCENE
    }


def load_risk_results():
    if not RISK_FILE.exists():
        raise FileNotFoundError(
            f"Risk result not found: {RISK_FILE}\n"
            f"Run the risk engine first."
        )

    with open(RISK_FILE) as f:
        data = json.load(f)

    if isinstance(data, list):
        records = data

    elif isinstance(data, dict):
        records = (
            data.get("results")
            or data.get("predictions")
            or data.get("records")
            or []
        )

    else:
        raise ValueError(
            "Unsupported risk JSON format"
        )

    return {
        record["uid"]: record
        for record in records
        if "uid" in record
    }


def parse_polygon_wkt(wkt):
    """
    Parse xView2 POLYGON WKT into pixel coordinates.

    Example:
    POLYGON ((x y, x y, ...))
    """

    if not wkt:
        return []

    match = re.search(
        r"POLYGON\s*\(\((.*?)\)\)",
        wkt,
    )

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


def draw_legend(draw):
    x = 30
    y = 30

    try:
        font = ImageFont.truetype(
            "/System/Library/Fonts/Helvetica.ttc",
            22,
        )
    except OSError:
        font = None

    title = "AEGIS AI — RESPONSE PRIORITY MAP"

    draw.text(
        (x, y),
        title,
        fill=(255, 255, 255),
        font=font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )

    y += 40

    for priority in PRIORITY_ORDER:
        color = PRIORITY_COLORS[priority]

        draw.rectangle(
            (x, y, x + 24, y + 24),
            fill=color,
            outline=(255, 255, 255),
            width=1,
        )

        draw.text(
            (x + 35, y),
            priority,
            fill=(255, 255, 255),
            font=font,
            stroke_width=2,
            stroke_fill=(0, 0, 0),
        )

        y += 32


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest = load_manifest()
    risk_results = load_risk_results()

    if not manifest:
        raise RuntimeError(
            f"No manifest rows found for scene: {SCENE}"
        )

    if not risk_results:
        raise RuntimeError(
            "No risk results found."
        )

    first_row = next(
        iter(manifest.values())
    )

    post_image_path = (
        PROJECT_ROOT
        / first_row["post_image"]
    )

    image = Image.open(
        post_image_path
    ).convert("RGB")

    overlay = Image.new(
        "RGBA",
        image.size,
        (0, 0, 0, 0),
    )

    draw = ImageDraw.Draw(overlay)

    counts = {
        priority: 0
        for priority in PRIORITY_ORDER
    }

    drawn = 0
    skipped = 0

    for uid, result in risk_results.items():

        row = manifest.get(uid)

        if row is None:
            skipped += 1
            continue

        priority = result.get(
            "response_priority"
        )

        if priority not in PRIORITY_COLORS:
            skipped += 1
            continue

        polygon = parse_polygon_wkt(
            row["polygon_wkt"]
        )

        if len(polygon) < 3:
            skipped += 1
            continue

        color = PRIORITY_COLORS[priority]

        # Transparent fill.
        fill_color = (
            *color,
            70,
        )

        # Strong outline.
        outline_color = (
            *color,
            235,
        )

        draw.polygon(
            polygon,
            fill=fill_color,
            outline=outline_color,
        )

        counts[priority] += 1
        drawn += 1

    result_image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB")

    legend = ImageDraw.Draw(
        result_image
    )

    draw_legend(legend)

    result_image.save(
        OUTPUT_FILE,
        quality=95,
    )

    print(
        "Aegis AI - Response Priority Map"
    )
    print("=" * 50)

    print(f"Scene:       {SCENE}")
    print(f"Buildings:   {len(risk_results)}")
    print(f"Mapped:      {drawn}")
    print(f"Skipped:     {skipped}")

    print()
    print("Response Priority")
    print("-" * 50)

    for priority in PRIORITY_ORDER:
        print(
            f"{priority:<10}: "
            f"{counts[priority]}"
        )

    print()
    print(
        f"Map saved: {OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()