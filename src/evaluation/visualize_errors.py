import csv
import re
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


PROJECT_ROOT = Path(__file__).resolve().parents[2]

ERRORS_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
    / "errors.csv"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "error_analysis"
    / "visualizations"
)

TOP_K = 5

FOCUS_PAIRS = [
    ("no-damage", "destroyed"),
    ("minor-damage", "major-damage"),
    ("major-damage", "destroyed"),
    ("destroyed", "major-damage"),
    ("minor-damage", "no-damage"),
]

CONTEXT_SIZE = 384
ZOOM_SIZE = 384
PADDING = 24


def load_errors():
    with open(ERRORS_FILE, "r", newline="") as f:
        return list(csv.DictReader(f))


def resolve_image_path(relative_path):
    return PROJECT_ROOT / relative_path


def parse_polygon_wkt(wkt):
    """
    Parse a simple POLYGON WKT string and return
    a list of (x, y) coordinate pairs.
    """
    if not wkt:
        raise ValueError("Empty polygon WKT")

    pairs = re.findall(
        r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)",
        wkt,
    )

    if len(pairs) < 3:
        raise ValueError(f"Could not parse polygon WKT: {wkt}")

    return [
        (float(x), float(y))
        for x, y in pairs
    ]


def polygon_bbox(polygon):
    xs = [point[0] for point in polygon]
    ys = [point[1] for point in polygon]

    return (
        min(xs),
        min(ys),
        max(xs),
        max(ys),
    )


def expand_bbox(bbox, image_size, padding):
    min_x, min_y, max_x, max_y = bbox

    width, height = image_size

    min_x = max(0, int(min_x - padding))
    min_y = max(0, int(min_y - padding))
    max_x = min(width, int(max_x + padding))
    max_y = min(height, int(max_y + padding))

    if max_x <= min_x:
        max_x = min(width, min_x + 1)

    if max_y <= min_y:
        max_y = min(height, min_y + 1)

    return (
        min_x,
        min_y,
        max_x,
        max_y,
    )


def crop_from_polygon(image, polygon, padding):
    bbox = polygon_bbox(polygon)
    crop_box = expand_bbox(
        bbox,
        image.size,
        padding,
    )

    crop = image.crop(crop_box)

    return crop, crop_box


def transform_polygon(polygon, crop_box):
    min_x, min_y, _, _ = crop_box

    return [
        (
            x - min_x,
            y - min_y,
        )
        for x, y in polygon
    ]


def resize_with_polygon(image, polygon, size):
    original_size = image.size

    image = image.resize(
        (size, size)
    )

    scale_x = size / original_size[0]
    scale_y = size / original_size[1]

    transformed = [
        (
            x * scale_x,
            y * scale_y,
        )
        for x, y in polygon
    ]

    return image, transformed


def draw_polygon(
    image,
    polygon,
    width=4,
):
    draw = ImageDraw.Draw(image)

    if len(polygon) >= 3:
        draw.line(
            polygon + [polygon[0]],
            fill="red",
            width=width,
        )


def make_difference(pre, post):
    if pre.size != post.size:
        post = post.resize(pre.size)

    pre_pixels = pre.load()
    post_pixels = post.load()

    diff = Image.new(
        "RGB",
        pre.size,
    )

    diff_pixels = diff.load()

    for y in range(pre.height):
        for x in range(pre.width):
            r1, g1, b1 = pre_pixels[x, y]
            r2, g2, b2 = post_pixels[x, y]

            diff_pixels[x, y] = (
                abs(r1 - r2),
                abs(g1 - g2),
                abs(b1 - b2),
            )

    return diff


def get_font(size):
    font_paths = [
        "/System/Library/Fonts/Helvetica.ttc",
        "/System/Library/Fonts/SFNS.ttf",
    ]

    for path in font_paths:
        try:
            return ImageFont.truetype(
                path,
                size,
            )
        except OSError:
            continue

    return ImageFont.load_default()


def add_panel_label(
    canvas,
    x,
    y,
    text,
    font,
):
    draw = ImageDraw.Draw(canvas)

    draw.text(
        (x, y),
        text,
        fill="black",
        font=font,
    )


def create_panel(row, output_path):
    pre_path = resolve_image_path(
        row["pre_image"]
    )

    post_path = resolve_image_path(
        row["post_image"]
    )

    pre_full = Image.open(
        pre_path
    ).convert("RGB")

    post_full = Image.open(
        post_path
    ).convert("RGB")

    polygon = parse_polygon_wkt(
        row["polygon_wkt"]
    )

    # --------------------------------------------------
    # Context crops
    # --------------------------------------------------

    pre_context, context_box = crop_from_polygon(
        pre_full,
        polygon,
        PADDING,
    )

    # IMPORTANT:
    # The same pre-disaster polygon bounding box is
    # used for the post-disaster image.
    post_context = post_full.crop(
        context_box
    )

    context_polygon = transform_polygon(
        polygon,
        context_box,
    )

    diff_context = make_difference(
        pre_context,
        post_context,
    )

    pre_context, context_polygon = (
        resize_with_polygon(
            pre_context,
            context_polygon,
            CONTEXT_SIZE,
        )
    )

    post_context, _ = resize_with_polygon(
        post_context,
        context_polygon,
        CONTEXT_SIZE,
    )

    diff_context, diff_polygon = (
        resize_with_polygon(
            diff_context,
            context_polygon,
            CONTEXT_SIZE,
        )
    )

    draw_polygon(
        pre_context,
        context_polygon,
    )

    draw_polygon(
        post_context,
        context_polygon,
    )

    draw_polygon(
        diff_context,
        diff_polygon,
    )

    # --------------------------------------------------
    # Zoom crops
    # --------------------------------------------------

    zoom_pre, zoom_box = crop_from_polygon(
        pre_full,
        polygon,
        PADDING // 2,
    )

    # Keep the exact same geographic/image box
    # for the post image.
    zoom_post = post_full.crop(
        zoom_box
    )

    zoom_polygon = transform_polygon(
        polygon,
        zoom_box,
    )

    zoom_diff = make_difference(
        zoom_pre,
        zoom_post,
    )

    zoom_pre, zoom_polygon = resize_with_polygon(
        zoom_pre,
        zoom_polygon,
        ZOOM_SIZE,
    )

    zoom_post, _ = resize_with_polygon(
        zoom_post,
        zoom_polygon,
        ZOOM_SIZE,
    )

    zoom_diff, zoom_diff_polygon = (
        resize_with_polygon(
            zoom_diff,
            zoom_polygon,
            ZOOM_SIZE,
        )
    )

    draw_polygon(
        zoom_pre,
        zoom_polygon,
        width=5,
    )

    draw_polygon(
        zoom_post,
        zoom_polygon,
        width=5,
    )

    draw_polygon(
        zoom_diff,
        zoom_diff_polygon,
        width=5,
    )

    # --------------------------------------------------
    # Canvas
    # --------------------------------------------------

    margin = 30
    header_height = 145
    row_gap = 70

    canvas_width = (
        margin * 4
        + CONTEXT_SIZE * 3
    )

    canvas_height = (
        header_height
        + CONTEXT_SIZE
        + row_gap
        + ZOOM_SIZE
        + 110
    )

    canvas = Image.new(
        "RGB",
        (
            canvas_width,
            canvas_height,
        ),
        "white",
    )

    draw = ImageDraw.Draw(canvas)

    title_font = get_font(30)
    info_font = get_font(21)
    small_font = get_font(15)
    label_font = get_font(21)

    # --------------------------------------------------
    # Header
    # --------------------------------------------------

    title = (
        f"{row['true_label']}  ->  "
        f"{row['predicted_label']}"
    )

    draw.text(
        (margin, 18),
        title,
        fill="black",
        font=title_font,
    )

    confidence = float(
        row["confidence"]
    )

    info = (
        f"Confidence: {confidence:.4f}    "
        f"Disaster: {row['disaster']}"
    )

    draw.text(
        (margin, 58),
        info,
        fill="black",
        font=info_font,
    )

    uid_text = (
        f"UID: {row['uid']}"
    )

    draw.text(
        (margin, 91),
        uid_text,
        fill="black",
        font=small_font,
    )

    draw.text(
        (
            margin,
            113,
        ),
        "Red outline = labeled building polygon",
        fill="black",
        font=small_font,
    )

    # --------------------------------------------------
    # Context row
    # --------------------------------------------------

    x_positions = [
        margin,
        margin * 2 + CONTEXT_SIZE,
        margin * 3 + CONTEXT_SIZE * 2,
    ]

    context_images = [
        pre_context,
        post_context,
        diff_context,
    ]

    context_labels = [
        "CONTEXT — PRE-DISASTER",
        "CONTEXT — POST-DISASTER",
        "CONTEXT — DIFFERENCE",
    ]

    context_y = header_height

    for x, image, label in zip(
        x_positions,
        context_images,
        context_labels,
    ):
        canvas.paste(
            image,
            (x, context_y),
        )

        add_panel_label(
            canvas,
            x,
            context_y + CONTEXT_SIZE + 8,
            label,
            label_font,
        )

    # --------------------------------------------------
    # Zoom row
    # --------------------------------------------------

    zoom_y = (
        context_y
        + CONTEXT_SIZE
        + row_gap
    )

    zoom_images = [
        zoom_pre,
        zoom_post,
        zoom_diff,
    ]

    zoom_labels = [
        "BUILDING ZOOM — PRE",
        "BUILDING ZOOM — POST",
        "BUILDING ZOOM — DIFFERENCE",
    ]

    for x, image, label in zip(
        x_positions,
        zoom_images,
        zoom_labels,
    ):
        canvas.paste(
            image,
            (x, zoom_y),
        )

        add_panel_label(
            canvas,
            x,
            zoom_y + ZOOM_SIZE + 8,
            label,
            label_font,
        )

    canvas.save(
        output_path,
        quality=95,
    )


def main():
    if not ERRORS_FILE.exists():
        raise FileNotFoundError(
            f"Errors file not found: {ERRORS_FILE}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = load_errors()

    generated = 0

    for true_label, predicted_label in FOCUS_PAIRS:
        matches = [
            row
            for row in rows
            if (
                row["true_label"] == true_label
                and row["predicted_label"]
                == predicted_label
            )
        ]

        matches.sort(
            key=lambda row: float(
                row["confidence"]
            ),
            reverse=True,
        )

        pair_dir = (
            OUTPUT_DIR
            / (
                f"{true_label.replace('-', '_')}"
                f"_to_"
                f"{predicted_label.replace('-', '_')}"
            )
        )

        pair_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for index, row in enumerate(
            matches[:TOP_K],
            start=1,
        ):
            filename = (
                f"{index:02d}_"
                f"{true_label.replace('-', '_')}"
                "_to_"
                f"{predicted_label.replace('-', '_')}"
                ".jpg"
            )

            output_path = (
                pair_dir / filename
            )

            create_panel(
                row,
                output_path,
            )

            generated += 1

    print("\nBuilding-aware error visualizations")
    print("-" * 55)
    print(f"Generated: {generated}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
