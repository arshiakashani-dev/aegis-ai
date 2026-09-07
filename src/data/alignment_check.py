from collections import defaultdict
from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageChops

from src.data.dataset import XView2BuildingDataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "alignment_check"

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

SAMPLES_PER_CLASS_PER_DISASTER = 50
WORST_EXAMPLES = 16


def disaster_from_scene(scene_id):
    return scene_id.rsplit("_", 1)[0]


def load_rgb(path):
    with Image.open(path) as image:
        return image.convert("RGB")


def to_gray_array(image):
    array = np.asarray(
        image,
        dtype=np.float32,
    ) / 255.0

    gray = (
        0.299 * array[:, :, 0]
        + 0.587 * array[:, :, 1]
        + 0.114 * array[:, :, 2]
    )

    return gray


def gradient_image(gray):
    dy, dx = np.gradient(gray)

    gradient = np.sqrt(
        dx * dx + dy * dy
    )

    gradient -= gradient.mean()

    height, width = gradient.shape

    window = np.outer(
        np.hanning(height),
        np.hanning(width),
    )

    return gradient * window


def estimate_translation(pre_image, post_image):
    pre = gradient_image(
        to_gray_array(pre_image)
    )

    post = gradient_image(
        to_gray_array(post_image)
    )

    fft_pre = np.fft.fft2(pre)
    fft_post = np.fft.fft2(post)

    cross_power = (
        fft_pre * np.conj(fft_post)
    )

    magnitude = np.abs(cross_power)

    cross_power /= (
        magnitude + 1e-8
    )

    correlation = np.fft.ifft2(
        cross_power
    ).real

    peak_y, peak_x = np.unravel_index(
        np.argmax(correlation),
        correlation.shape,
    )

    height, width = correlation.shape

    if peak_x > width // 2:
        peak_x -= width

    if peak_y > height // 2:
        peak_y -= height

    dx = int(peak_x)
    dy = int(peak_y)

    shift = float(
        np.sqrt(dx * dx + dy * dy)
    )

    peak_score = float(
        correlation.max()
    )

    return dx, dy, shift, peak_score


def select_samples(dataset):
    groups = defaultdict(list)

    for index, row in enumerate(dataset.rows):
        disaster = disaster_from_scene(
            row["scene_id"]
        )

        damage = row["damage_label"]

        key = (
            disaster,
            damage,
        )

        groups[key].append(index)

    rng = np.random.default_rng(42)

    selected = []

    for key in sorted(groups):
        indices = groups[key]

        count = min(
            len(indices),
            SAMPLES_PER_CLASS_PER_DISASTER,
        )

        if len(indices) > count:
            indices = rng.choice(
                indices,
                size=count,
                replace=False,
            ).tolist()

        selected.extend(indices)

    return sorted(selected)


def analyze(dataset, indices):
    results = []

    for number, index in enumerate(
        indices,
        start=1,
    ):
        row = dataset.rows[index]

        scene_id = row["scene_id"]
        uid = row["uid"]
        damage = row["damage_label"]

        disaster = disaster_from_scene(
            scene_id
        )

        sample_dir = (
            dataset.cache_dir
            / scene_id
        )

        pre_path = (
            sample_dir
            / f"{uid}_pre.jpg"
        )

        post_path = (
            sample_dir
            / f"{uid}_post.jpg"
        )

        pre_image = load_rgb(pre_path)
        post_image = load_rgb(post_path)

        dx, dy, shift, peak_score = (
            estimate_translation(
                pre_image,
                post_image,
            )
        )

        results.append(
            {
                "index": index,
                "scene_id": scene_id,
                "uid": uid,
                "disaster": disaster,
                "damage_label": damage,
                "dx": dx,
                "dy": dy,
                "shift_pixels": shift,
                "peak_score": peak_score,
                "pre_path": pre_path,
                "post_path": post_path,
            }
        )

        if (
            number % 250 == 0
            or number == len(indices)
        ):
            print(
                f"Processed "
                f"{number}/{len(indices)}"
            )

    return results


def print_shift_summary(
    title,
    results,
):
    if not results:
        return

    shifts = np.array(
        [
            result["shift_pixels"]
            for result in results
        ],
        dtype=np.float32,
    )

    print()
    print(title)
    print("-" * 70)

    print(
        f"Samples       : {len(shifts)}"
    )

    print(
        f"Median shift  : "
        f"{np.median(shifts):.2f} px"
    )

    print(
        f"75th percentile: "
        f"{np.percentile(shifts, 75):.2f} px"
    )

    print(
        f"90th percentile: "
        f"{np.percentile(shifts, 90):.2f} px"
    )

    print(
        f"95th percentile: "
        f"{np.percentile(shifts, 95):.2f} px"
    )

    print(
        f"Max shift     : "
        f"{shifts.max():.2f} px"
    )

    within_2 = (
        shifts <= 2
    ).mean() * 100

    within_5 = (
        shifts <= 5
    ).mean() * 100

    above_5 = (
        shifts > 5
    ).mean() * 100

    print(
        f"<= 2 px       : "
        f"{within_2:.1f}%"
    )

    print(
        f"<= 5 px       : "
        f"{within_5:.1f}%"
    )

    print(
        f"> 5 px        : "
        f"{above_5:.1f}%"
    )


def save_csv(results):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_file = (
        OUTPUT_DIR
        / "alignment_summary.csv"
    )

    fields = [
        "scene_id",
        "uid",
        "disaster",
        "damage_label",
        "dx",
        "dy",
        "shift_pixels",
        "peak_score",
    ]

    with open(
        output_file,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for result in results:
            writer.writerow(
                {
                    field: result[field]
                    for field in fields
                }
            )

    print()
    print(
        "Saved:",
        output_file,
    )


def save_worst_grid(results):
    no_damage_results = [
        result
        for result in results
        if result["damage_label"]
        == "no-damage"
    ]

    worst = sorted(
        no_damage_results,
        key=lambda item: item[
            "shift_pixels"
        ],
        reverse=True,
    )[:WORST_EXAMPLES]

    if not worst:
        return

    fig, axes = plt.subplots(
        len(worst),
        4,
        figsize=(
            12,
            len(worst) * 3,
        ),
    )

    if len(worst) == 1:
        axes = [axes]

    for row_number, result in enumerate(
        worst
    ):
        pre_image = load_rgb(
            result["pre_path"]
        )

        post_image = load_rgb(
            result["post_path"]
        )

        overlay = Image.blend(
            pre_image,
            post_image,
            alpha=0.5,
        )

        difference = ImageChops.difference(
            pre_image,
            post_image,
        )

        axes[row_number][0].imshow(
            pre_image
        )
        axes[row_number][0].set_title(
            "PRE"
        )

        axes[row_number][1].imshow(
            post_image
        )
        axes[row_number][1].set_title(
            "POST"
        )

        axes[row_number][2].imshow(
            overlay
        )
        axes[row_number][2].set_title(
            "OVERLAY"
        )

        axes[row_number][3].imshow(
            difference
        )
        axes[row_number][3].set_title(
            "DIFFERENCE"
        )

        for axis in axes[row_number]:
            axis.axis("off")

        axes[row_number][0].set_ylabel(
            (
                f"{result['disaster']}\n"
                f"dx={result['dx']} "
                f"dy={result['dy']}\n"
                f"shift="
                f"{result['shift_pixels']:.1f}px"
            ),
            fontsize=8,
        )

    fig.suptitle(
        "Worst estimated alignment - "
        "No-damage validation samples",
        fontsize=14,
    )

    plt.tight_layout()

    output_file = (
        OUTPUT_DIR
        / "worst_no_damage_alignment.png"
    )

    fig.savefig(
        output_file,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        "Saved:",
        output_file,
    )


def main():
    print()
    print(
        "Aegis AI - Alignment Quality Check"
    )
    print("=" * 70)

    dataset = XView2BuildingDataset(
        VAL_FILE
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    indices = select_samples(
        dataset
    )

    print(
        "Alignment samples:",
        len(indices),
    )

    results = analyze(
        dataset,
        indices,
    )

    print_shift_summary(
        "ALL SAMPLED BUILDINGS",
        results,
    )

    no_damage_results = [
        result
        for result in results
        if result["damage_label"]
        == "no-damage"
    ]

    print_shift_summary(
        "NO-DAMAGE SAMPLES "
        "(PRIMARY ALIGNMENT CHECK)",
        no_damage_results,
    )

    for class_name in CLASS_NAMES:
        class_results = [
            result
            for result in results
            if result["damage_label"]
            == class_name
        ]

        print_shift_summary(
            f"CLASS: {class_name}",
            class_results,
        )

    save_csv(results)

    save_worst_grid(results)

    print()
    print("=" * 70)
    print(
        "Alignment quality check complete."
    )
    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()