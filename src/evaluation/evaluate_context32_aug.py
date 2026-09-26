import csv
import json
from collections import defaultdict
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


def evaluate(model, loader, device):
    all_predictions = []
    all_labels = []
    all_scene_ids = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            predictions = outputs.argmax(dim=1)

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                labels.cpu().tolist()
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


def save_json(data, path):
    with open(path, "w") as f:
        json.dump(
            data,
            f,
            indent=2,
        )


def save_confusion_matrix(matrix, path):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(
        figsize=(8, 7)
    )

    image = ax.imshow(matrix)

    ax.set_title(
        "Aegis AI — Context32 + Augmentation"
    )

    ax.set_xlabel(
        "Predicted label"
    )

    ax.set_ylabel(
        "True label"
    )

    ax.set_xticks(
        range(len(CLASS_NAMES))
    )

    ax.set_yticks(
        range(len(CLASS_NAMES))
    )

    ax.set_xticklabels(
        CLASS_NAMES,
        rotation=30,
        ha="right",
    )

    ax.set_yticklabels(
        CLASS_NAMES
    )

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(
        image,
        ax=ax,
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)


def calculate_per_disaster(
    labels,
    predictions,
    scene_ids,
):
    grouped_labels = defaultdict(list)
    grouped_predictions = defaultdict(list)

    for label, prediction, scene_id in zip(
        labels,
        predictions,
        scene_ids,
    ):
        disaster = scene_id.split("_")[0]

        grouped_labels[disaster].append(
            label
        )

        grouped_predictions[disaster].append(
            prediction
        )

    results = []

    for disaster in sorted(grouped_labels):
        y_true = grouped_labels[disaster]
        y_pred = grouped_predictions[disaster]

        results.append(
            {
                "disaster": disaster,
                "samples": len(y_true),
                "accuracy": accuracy_score(
                    y_true,
                    y_pred,
                ),
                "macro_f1": f1_score(
                    y_true,
                    y_pred,
                    average="macro",
                    zero_division=0,
                ),
                "weighted_f1": f1_score(
                    y_true,
                    y_pred,
                    average="weighted",
                    zero_division=0,
                ),
            }
        )

    return results


def save_per_disaster_csv(
    results,
    path,
):
    fieldnames = [
        "disaster",
        "samples",
        "accuracy",
        "macro_f1",
        "weighted_f1",
    ]

    with open(
        path,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()

        for row in results:
            writer.writerow(row)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(
        "Aegis AI - Context32 + Augmentation Evaluation"
    )

    print()
    print("=" * 55)

    print(
        f"Device: {device}"
    )

    print(
        f"Checkpoint: {CHECKPOINT}"
    )

    print(
        f"Validation manifest: {VAL_FILE}"
    )

    dataset, loader = load_validation_data()

    print(
        f"Validation samples: {len(dataset)}"
    )

    model = load_model(device)

    (
        labels,
        predictions,
        scene_ids,
    ) = evaluate(
        model,
        loader,
        device,
    )

    accuracy = accuracy_score(
        labels,
        predictions,
    )

    macro_f1 = f1_score(
        labels,
        predictions,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        labels,
        predictions,
        average="weighted",
        zero_division=0,
    )

    report = classification_report(
        labels,
        predictions,
        target_names=CLASS_NAMES,
        output_dict=True,
        zero_division=0,
    )

    matrix = confusion_matrix(
        labels,
        predictions,
    )

    print()
    print("Overall metrics")
    print("-" * 55)

    print(
        f"Accuracy:    {accuracy:.4f}"
    )

    print(
        f"Macro F1:    {macro_f1:.4f}"
    )

    print(
        f"Weighted F1: {weighted_f1:.4f}"
    )

    print()
    print("Classification report")
    print("-" * 55)

    print(
        classification_report(
            labels,
            predictions,
            target_names=CLASS_NAMES,
            zero_division=0,
        )
    )

    print("Confusion matrix")
    print("-" * 55)

    print(matrix)

    confusion_matrix_path = (
        OUTPUT_DIR
        / "confusion_matrix.png"
    )

    save_confusion_matrix(
        matrix,
        confusion_matrix_path,
    )

    per_disaster = calculate_per_disaster(
        labels,
        predictions,
        scene_ids,
    )

    print()
    print("Per-disaster results")
    print("-" * 55)

    for result in per_disaster:
        print(
            f"{result['disaster']} "
            f"samples={result['samples']} "
            f"accuracy={result['accuracy']:.4f} "
            f"macro_f1={result['macro_f1']:.4f}"
        )

    metrics = {
        "model": "AegisResNet18ChangeAware",
        "experiment": "context32_aug",
        "device": str(device),
        "validation_samples": len(dataset),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
    }

    metrics_path = (
        OUTPUT_DIR
        / "metrics.json"
    )

    save_json(
        metrics,
        metrics_path,
    )

    per_disaster_path = (
        OUTPUT_DIR
        / "per_disaster.csv"
    )

    save_per_disaster_csv(
        per_disaster,
        per_disaster_path,
    )

    print()
    print("Evaluation complete.")
    print()
    print(
        f"Outputs: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()