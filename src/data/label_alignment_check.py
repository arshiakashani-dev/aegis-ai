import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "xview2"
    / "train"
    / "train"
    / "labels"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "alignment_check"
)

MAX_SAMPLES_PER_SCENE = 100


def disaster_from_scene(scene_id):
    return scene_id.rsplit("_", 1)[0]


def polygon_points_from_wkt(wkt):
    """
    Extract polygon exterior coordinates from WKT.

    Example:
    POLYGON ((x y, x y, ...))
    """
    match = re.search(
        r"POLYGON\s*\(\((.*?)\)\)",
        wkt,
        flags=re.IGNORECASE,
    )

    if not match:
        return []

    points = []

    for pair in match.group(1).split(","):
        values = pair.strip().split()

        if len(values) < 2:
            continue

        x = float(values[0])
        y = float(values[1])

        points.append((x, y))

    return points


def polygon_centroid(points):
    """
    Area-weighted polygon centroid.

    Falls back to mean vertex position for degenerate polygons.
    """
    if len(points) < 3:
        if not points:
            return None

        array = np.asarray(
            points,
            dtype=np.float64,
        )

        return (
            float(array[:, 0].mean()),
            float(array[:, 1].mean()),
        )

    points = list(points)

    if points[0] != points[-1]:
        points.append(points[0])

    area_twice = 0.0
    centroid_x = 0.0
    centroid_y = 0.0

    for i in range(len(points) - 1):
        x1, y1 = points[i]
        x2, y2 = points[i + 1]

        cross = (
            x1 * y2
            - x2 * y1
        )

        area_twice += cross

        centroid_x += (
            x1 + x2
        ) * cross

        centroid_y += (
            y1 + y2
        ) * cross

    if abs(area_twice) < 1e-8:
        array = np.asarray(
            points[:-1],
            dtype=np.float64,
        )

        return (
            float(array[:, 0].mean()),
            float(array[:, 1].mean()),
        )

    centroid_x /= (
        3.0 * area_twice
    )

    centroid_y /= (
        3.0 * area_twice
    )

    return (
        centroid_x,
        centroid_y,
    )


def load_label_file(path):
    with open(
        path,
        "r",
    ) as file:
        return json.load(file)


def extract_buildings(data):
    buildings = {}

    features = (
        data
        .get("features", {})
        .get("xy", [])
    )

    for feature in features:
        properties = feature.get(
            "properties",
            {},
        )

        if properties.get(
            "feature_type"
        ) != "building":
            continue

        uid = properties.get("uid")
        wkt = feature.get("wkt")

        if not uid or not wkt:
            continue

        points = polygon_points_from_wkt(
            wkt
        )

        centroid = polygon_centroid(
            points
        )

        if centroid is None:
            continue

        buildings[uid] = {
            "centroid": centroid,
            "subtype": properties.get(
                "subtype"
            ),
        }

    return buildings


def discover_scenes():
    pre_files = sorted(
        LABEL_DIR.glob(
            "*_pre_disaster.json"
        )
    )

    post_files = sorted(
        LABEL_DIR.glob(
            "*_post_disaster.json"
        )
    )

    pre_by_scene = {}
    post_by_scene = {}

    for path in pre_files:
        scene_id = path.name.replace(
            "_pre_disaster.json",
            "",
        )

        pre_by_scene[scene_id] = path

    for path in post_files:
        scene_id = path.name.replace(
            "_post_disaster.json",
            "",
        )

        post_by_scene[scene_id] = path

    scenes = sorted(
        set(pre_by_scene)
        & set(post_by_scene)
    )

    return [
        (
            scene,
            pre_by_scene[scene],
            post_by_scene[scene],
        )
        for scene in scenes
    ]


def analyze_scene(
    scene_id,
    pre_path,
    post_path,
):
    pre_data = load_label_file(
        pre_path
    )

    post_data = load_label_file(
        post_path
    )

    pre_buildings = extract_buildings(
        pre_data
    )

    post_buildings = extract_buildings(
        post_data
    )

    shared_uids = sorted(
        set(pre_buildings)
        & set(post_buildings)
    )

    if (
        MAX_SAMPLES_PER_SCENE
        and len(shared_uids)
        > MAX_SAMPLES_PER_SCENE
    ):
        rng = np.random.default_rng(
            42
        )

        shared_uids = rng.choice(
            shared_uids,
            size=MAX_SAMPLES_PER_SCENE,
            replace=False,
        ).tolist()

    samples = []

    for uid in shared_uids:
        pre_centroid = np.asarray(
            pre_buildings[uid][
                "centroid"
            ],
            dtype=np.float64,
        )

        post_centroid = np.asarray(
            post_buildings[uid][
                "centroid"
            ],
            dtype=np.float64,
        )

        displacement = (
            post_centroid
            - pre_centroid
        )

        magnitude = float(
            np.linalg.norm(
                displacement
            )
        )

        samples.append(
            {
                "scene_id": scene_id,
                "uid": uid,
                "disaster": disaster_from_scene(
                    scene_id
                ),
                "damage_label": (
                    post_buildings[uid][
                        "subtype"
                    ]
                ),
                "dx": float(
                    displacement[0]
                ),
                "dy": float(
                    displacement[1]
                ),
                "shift_pixels": magnitude,
            }
        )

    return samples


def summarize(
    title,
    samples,
):
    if not samples:
        print()
        print(title)
        print("-" * 70)
        print("No samples.")
        return

    shifts = np.asarray(
        [
            sample["shift_pixels"]
            for sample in samples
        ],
        dtype=np.float64,
    )

    dx = np.asarray(
        [
            sample["dx"]
            for sample in samples
        ],
        dtype=np.float64,
    )

    dy = np.asarray(
        [
            sample["dy"]
            for sample in samples
        ],
        dtype=np.float64,
    )

    print()
    print(title)
    print("-" * 70)

    print(
        f"Samples        : {len(samples)}"
    )

    print(
        f"Median |shift| : "
        f"{np.median(shifts):.2f} px"
    )

    print(
        f"Mean |shift|   : "
        f"{np.mean(shifts):.2f} px"
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
        f"Max |shift|    : "
        f"{np.max(shifts):.2f} px"
    )

    print(
        f"Median dx      : "
        f"{np.median(dx):.2f} px"
    )

    print(
        f"Median dy      : "
        f"{np.median(dy):.2f} px"
    )

    print(
        f"<= 2 px        : "
        f"{(shifts <= 2).mean() * 100:.1f}%"
    )

    print(
        f"<= 5 px        : "
        f"{(shifts <= 5).mean() * 100:.1f}%"
    )

    print(
        f"> 5 px         : "
        f"{(shifts > 5).mean() * 100:.1f}%"
    )


def compute_scene_translation(
    samples,
):
    """
    Estimate the dominant translation of a scene
    using the median dx/dy across shared buildings.
    """

    if not samples:
        return 0.0, 0.0

    dx = np.median(
        [
            sample["dx"]
            for sample in samples
        ]
    )

    dy = np.median(
        [
            sample["dy"]
            for sample in samples
        ]
    )

    return float(dx), float(dy)


def add_residuals(
    scene_samples,
):
    scene_dx, scene_dy = (
        compute_scene_translation(
            scene_samples
        )
    )

    for sample in scene_samples:
        residual_dx = (
            sample["dx"]
            - scene_dx
        )

        residual_dy = (
            sample["dy"]
            - scene_dy
        )

        residual = np.sqrt(
            residual_dx ** 2
            + residual_dy ** 2
        )

        sample[
            "scene_dx"
        ] = scene_dx

        sample[
            "scene_dy"
        ] = scene_dy

        sample[
            "residual_shift"
        ] = float(residual)


def main():
    print()
    print(
        "Aegis AI - Label Alignment Check"
    )
    print("=" * 70)

    if not LABEL_DIR.exists():
        raise FileNotFoundError(
            f"Label directory not found:\n"
            f"{LABEL_DIR}"
        )

    scenes = discover_scenes()

    print(
        "Paired scenes:",
        len(scenes),
    )

    all_samples = []
    scene_results = []

    for number, (
        scene_id,
        pre_path,
        post_path,
    ) in enumerate(
        scenes,
        start=1,
    ):
        samples = analyze_scene(
            scene_id,
            pre_path,
            post_path,
        )

        if not samples:
            continue

        add_residuals(
            samples
        )

        all_samples.extend(
            samples
        )

        scene_results.append(
            {
                "scene_id": scene_id,
                "disaster": disaster_from_scene(
                    scene_id
                ),
                "building_count": len(
                    samples
                ),
                "scene_dx": samples[0][
                    "scene_dx"
                ],
                "scene_dy": samples[0][
                    "scene_dy"
                ],
                "median_shift": float(
                    np.median(
                        [
                            sample[
                                "shift_pixels"
                            ]
                            for sample in samples
                        ]
                    )
                ),
                "median_residual": float(
                    np.median(
                        [
                            sample[
                                "residual_shift"
                            ]
                            for sample in samples
                        ]
                    )
                ),
            }
        )

        if (
            number % 250 == 0
            or number == len(scenes)
        ):
            print(
                f"Processed "
                f"{number}/{len(scenes)} scenes"
            )

    print()
    print(
        "Matched building samples:",
        len(all_samples),
    )

    summarize(
        "RAW BUILDING CENTROID SHIFT",
        all_samples,
    )

    residuals = [
        sample
        for sample in all_samples
        if "residual_shift" in sample
    ]

    if residuals:
        residual_values = np.asarray(
            [
                sample[
                    "residual_shift"
                ]
                for sample in residuals
            ],
            dtype=np.float64,
        )

        print()
        print(
            "AFTER REMOVING "
            "SCENE-LEVEL TRANSLATION"
        )
        print("-" * 70)

        print(
            f"Samples        : "
            f"{len(residual_values)}"
        )

        print(
            f"Median residual: "
            f"{np.median(residual_values):.2f} px"
        )

        print(
            f"75th percentile: "
            f"{np.percentile(residual_values, 75):.2f} px"
        )

        print(
            f"90th percentile: "
            f"{np.percentile(residual_values, 90):.2f} px"
        )

        print(
            f"95th percentile: "
            f"{np.percentile(residual_values, 95):.2f} px"
        )

        print(
            f"<= 2 px        : "
            f"{(residual_values <= 2).mean() * 100:.1f}%"
        )

        print(
            f"<= 5 px        : "
            f"{(residual_values <= 5).mean() * 100:.1f}%"
        )

        print(
            f"> 5 px         : "
            f"{(residual_values > 5).mean() * 100:.1f}%"
        )

    for class_name in [
        "no-damage",
        "minor-damage",
        "major-damage",
        "destroyed",
    ]:
        class_samples = [
            sample
            for sample in all_samples
            if sample["damage_label"]
            == class_name
        ]

        summarize(
            f"CLASS: {class_name}",
            class_samples,
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    building_output = (
        OUTPUT_DIR
        / "label_alignment_buildings.csv"
    )

    with open(
        building_output,
        "w",
    ) as file:
        file.write(
            "scene_id,uid,disaster,"
            "damage_label,dx,dy,"
            "shift_pixels,"
            "scene_dx,scene_dy,"
            "residual_shift\n"
        )

        for sample in all_samples:
            file.write(
                f"{sample['scene_id']},"
                f"{sample['uid']},"
                f"{sample['disaster']},"
                f"{sample['damage_label']},"
                f"{sample['dx']:.4f},"
                f"{sample['dy']:.4f},"
                f"{sample['shift_pixels']:.4f},"
                f"{sample['scene_dx']:.4f},"
                f"{sample['scene_dy']:.4f},"
                f"{sample['residual_shift']:.4f}\n"
            )

    scene_output = (
        OUTPUT_DIR
        / "label_alignment_scenes.csv"
    )

    with open(
        scene_output,
        "w",
    ) as file:
        file.write(
            "scene_id,disaster,"
            "building_count,"
            "scene_dx,scene_dy,"
            "median_shift,"
            "median_residual\n"
        )

        for result in scene_results:
            file.write(
                f"{result['scene_id']},"
                f"{result['disaster']},"
                f"{result['building_count']},"
                f"{result['scene_dx']:.4f},"
                f"{result['scene_dy']:.4f},"
                f"{result['median_shift']:.4f},"
                f"{result['median_residual']:.4f}\n"
            )

    print()
    print(
        "Saved:",
        building_output,
    )

    print(
        "Saved:",
        scene_output,
    )

    print()
    print(
        "Label alignment check complete."
    )


if __name__ == "__main__":
    main()