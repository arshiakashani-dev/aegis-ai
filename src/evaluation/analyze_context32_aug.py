import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from torch.utils.data import DataLoader

from src.data.context32_change_dataset import XView2Context32Dataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"

CHECKPOINT = (
    PROJECT_ROOT
    / "checkpoints"
    / "context32_aug_resnet_change_aware.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation"
    / "context32_aug"
)

BATCH_SIZE = 16
NUM_WORKERS = 0

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_model(device):
    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_validation_data():
    dataset = XView2Context32Dataset(
        str(VAL_FILE)
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    return dataset, loader


def get_disaster_name(scene_id):
    if scene_id.startswith("hurricane-"):
        parts = scene_id.split("-")

        if len(parts) >= 2:
            return "-".join(parts[:2])

    return scene_id.split("_")[0]


def evaluate(model, loader, device):
    all_labels = []
    all_predictions = []
    all_scene_ids = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_labels.extend(
                labels.cpu().tolist()
            )

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            batch_size = len(labels)

            for i in range(batch_size):
                row_index = sample_index + i

                scene_id = (
                    loader.dataset.rows[row_index][
                        "scene_id"
                    ]
                )

                all_scene_ids.append(scene_id)

            sample_index += batch_size

    return (
        all_labels,
        all_predictions,
        all_scene_ids,
    )


def build_disaster_groups(
    labels,
    predictions,
    scene_ids,
):
    groups = defaultdict(
        lambda: {
            "labels": [],
            "predictions": [],
        }
    )

    for label, prediction, scene_id in zip(
        labels,
        predictions,
        scene_ids,
    ):
        disaster = get_disaster_name(
            scene_id
        )

        groups[disaster]["labels"].append(
            label
        )

        groups[disaster]["predictions"].append(
            prediction
        )

    return groups


def calculate_overall_class_distribution(
    labels,
):
    counts = Counter(labels)

    total = len(labels)

    results = []

    for class_index, class_name in enumerate(
        CLASS_NAMES
    ):
        count = counts[class_index]

        percentage = (
            count / total * 100
            if total
            else 0.0
        )

        results.append(
            {
                "class": class_name,
                "count": count,
                "percentage": percentage,
            }
        )

    return results


def calculate_disaster_class_distribution(
    groups,
):
    results = []

    for disaster in sorted(groups):
        labels = groups[disaster]["labels"]

        total = len(labels)

        counts = Counter(labels)

        row = {
            "disaster": disaster,
            "samples": total,
        }

        for class_index, class_name in enumerate(
            CLASS_NAMES
        ):
            count = counts[class_index]

            percentage = (
                count / total * 100
                if total
                else 0.0
            )

            row[f"{class_name}_count"] = count
            row[f"{class_name}_percentage"] = (
                percentage
            )

        results.append(row)

    return results


def calculate_disaster_class_metrics(
    groups,
):
    results = []

    for disaster in sorted(groups):
        y_true = groups[disaster]["labels"]
        y_pred = groups[disaster]["predictions"]

        report = classification_report(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
            target_names=CLASS_NAMES,
            output_dict=True,
            zero_division=0,
        )

        row = {
            "disaster": disaster,
            "samples": len(y_true),
            "accuracy": accuracy_score(
                y_true,
                y_pred,
            ),
        }

        for class_name in CLASS_NAMES:
            row[
                f"{class_name}_precision"
            ] = report[class_name]["precision"]

            row[
                f"{class_name}_recall"
            ] = report[class_name]["recall"]

            row[
                f"{class_name}_f1"
            ] = report[class_name]["f1-score"]

            row[
                f"{class_name}_support"
            ] = report[class_name]["support"]

        row["macro_f1"] = f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        )

        row["weighted_f1"] = f1_score(
            y_true,
            y_pred,
            average="weighted",
            zero_division=0,
        )

        results.append(row)

    return results


def calculate_confusion_matrices(
    groups,
):
    results = {}

    for disaster in sorted(groups):
        y_true = groups[disaster]["labels"]
        y_pred = groups[disaster]["predictions"]

        matrix = confusion_matrix(
            y_true,
            y_pred,
            labels=list(range(len(CLASS_NAMES))),
        )

        results[disaster] = matrix.tolist()

    return results


def save_json(data, path):
    with open(
        path,
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def save_csv(rows, path):
    if not rows:
        return

    fieldnames = list(rows[0].keys())

    with open(
        path,
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(rows)


def print_class_distribution(
    results,
):
    print(
        "\nOverall class distribution"
    )

    print(
        "-" * 55
    )

    for row in results:
        print(
            f"{row['class']:<15}"
            f" samples={row['count']:<7}"
            f" percentage={row['percentage']:.2f}%"
        )


def print_disaster_distribution(
    results,
):
    print(
        "\nPer-disaster class distribution"
    )

    print(
        "-" * 100
    )

    for row in results:
        print(
            f"\n{row['disaster']}"
            f"  samples={row['samples']}"
        )

        for class_name in CLASS_NAMES:
            count = row[
                f"{class_name}_count"
            ]

            percentage = row[
                f"{class_name}_percentage"
            ]

            print(
                f"  {class_name:<15}"
                f" {count:<7}"
                f" ({percentage:6.2f}%)"
            )


def print_disaster_metrics(
    results,
):
    print(
        "\nPer-disaster class metrics"
    )

    print(
        "-" * 100
    )

    for row in results:
        print(
            f"\n{row['disaster']}"
            f"  samples={row['samples']}"
            f"  accuracy={row['accuracy']:.4f}"
            f"  macro_f1={row['macro_f1']:.4f}"
            f"  weighted_f1={row['weighted_f1']:.4f}"
        )

        for class_name in CLASS_NAMES:
            precision = row[
                f"{class_name}_precision"
            ]

            recall = row[
                f"{class_name}_recall"
            ]

            f1 = row[
                f"{class_name}_f1"
            ]

            support = row[
                f"{class_name}_support"
            ]

            print(
                f"  {class_name:<15}"
                f" P={precision:.3f}"
                f" R={recall:.3f}"
                f" F1={f1:.3f}"
                f" support={support}"
            )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        "Aegis AI - Context32 + Augmentation Analysis"
    )

    print(
        "=" * 55
    )

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Validation manifest: {VAL_FILE}"
    )

    model = load_model(device)

    dataset, loader = load_validation_data()

    print(
        f"Validation samples: {len(dataset)}"
    )

    labels, predictions, scene_ids = evaluate(
        model,
        loader,
        device,
    )

    groups = build_disaster_groups(
        labels,
        predictions,
        scene_ids,
    )

    overall_distribution = (
        calculate_overall_class_distribution(
            labels
        )
    )

    disaster_distribution = (
        calculate_disaster_class_distribution(
            groups
        )
    )

    disaster_metrics = (
        calculate_disaster_class_metrics(
            groups
        )
    )

    confusion_matrices = (
        calculate_confusion_matrices(
            groups
        )
    )

    print_class_distribution(
        overall_distribution
    )

    print_disaster_distribution(
        disaster_distribution
    )

    print_disaster_metrics(
        disaster_metrics
    )

    save_json(
        overall_distribution,
        OUTPUT_DIR
        / "overall_class_distribution.json",
    )

    save_csv(
        disaster_distribution,
        OUTPUT_DIR
        / "per_disaster_class_distribution.csv",
    )

    save_csv(
        disaster_metrics,
        OUTPUT_DIR
        / "per_disaster_class_metrics.csv",
    )

    save_json(
        confusion_matrices,
        OUTPUT_DIR
        / "per_disaster_confusion_matrices.json",
    )

    print(
        "\nAnalysis complete."
    )

    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()