from pathlib import Path
import argparse
import csv
import json

import torch
from PIL import Image

from src.inference.predict import (
    CLASS_NAMES,
    build_input,
    get_device,
    load_model,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"


def load_scene_rows(split, scene_id):
    manifest_path = PROCESSED_DIR / f"{split}.csv"

    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    with open(
        manifest_path,
        "r",
        newline="",
    ) as file:
        rows = list(csv.DictReader(file))

    scene_rows = [
        row
        for row in rows
        if row["scene_id"] == scene_id
    ]

    if not scene_rows:
        raise ValueError(
            f"Scene not found in {split}.csv: {scene_id}"
        )

    return scene_rows


def predict_scene(split, scene_id):
    device = get_device()
    model = load_model(device)

    rows = load_scene_rows(
        split,
        scene_id,
    )

    results = []

    print()
    print("Aegis AI - Scene Inference")
    print("=" * 50)
    print("Scene:", scene_id)
    print("Split:", split)
    print("Buildings:", len(rows))
    print("Device:", device)
    print()

    for index, row in enumerate(rows, start=1):
        uid = row["uid"]

        scene_dir = (
            PROCESSED_DIR
            / "crops"
            / split
            / scene_id
        )

        pre_path = (
            scene_dir
            / f"{uid}_pre.jpg"
        )

        post_path = (
            scene_dir
            / f"{uid}_post.jpg"
        )

        if not pre_path.exists():
            raise FileNotFoundError(pre_path)

        if not post_path.exists():
            raise FileNotFoundError(post_path)

        image = build_input(
            pre_path,
            post_path,
        )

        image = image.unsqueeze(0).to(device)

        with torch.inference_mode():
            logits = model(image)

            probabilities = torch.softmax(
                logits,
                dim=1,
            )[0]

        predicted_class = int(
            torch.argmax(probabilities).item()
        )

        prediction = CLASS_NAMES[predicted_class]

        confidence = float(
            probabilities[predicted_class].item()
        )

        true_label = row["damage_label"]

        results.append(
            {
                "scene_id": scene_id,
                "uid": uid,
                "true_label": true_label,
                "prediction": prediction,
                "confidence": confidence,
                "probabilities": {
                    name: float(
                        probabilities[class_id].item()
                    )
                    for class_id, name
                    in enumerate(CLASS_NAMES)
                },
            }
        )

        if index <= 10:
            print(
                f"{index:4d}. "
                f"{uid[:8]}... "
                f"true={true_label:12s} "
                f"pred={prediction:12s} "
                f"conf={confidence:.4f}"
            )

    return results


def print_summary(results):
    counts = {
        name: 0
        for name in CLASS_NAMES
    }

    for result in results:
        counts[result["prediction"]] += 1

    print()
    print("Predicted Damage Summary")
    print("-" * 50)

    for name in CLASS_NAMES:
        print(
            f"{name:12s}: {counts[name]}"
        )

    correct = sum(
        result["true_label"]
        == result["prediction"]
        for result in results
    )

    accuracy = (
        correct / len(results)
        if results
        else 0.0
    )

    print()
    print(
        f"Scene accuracy: "
        f"{accuracy:.4f}"
    )


def save_results(
    results,
    split,
    scene_id,
):
    output_dir = (
        PROJECT_ROOT
        / "outputs"
        / "scene_inference"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = (
        output_dir
        / f"{scene_id}.json"
    )

    with open(
        output_path,
        "w",
    ) as file:
        json.dump(
            results,
            file,
            indent=2,
        )

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aegis AI scene-level "
            "building damage inference"
        )
    )

    parser.add_argument(
        "--split",
        default="val",
        choices=["train", "val"],
    )

    parser.add_argument(
        "--scene",
        required=True,
        help="xView2 scene ID",
    )

    args = parser.parse_args()

    results = predict_scene(
        args.split,
        args.scene,
    )

    print_summary(results)

    output_path = save_results(
        results,
        args.split,
        args.scene,
    )

    print()
    print(
        "Results saved:",
        output_path,
    )


if __name__ == "__main__":
    main()
