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

from src.data.change_dataset import XView2ChangeDataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_resnet_change_aware.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"

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

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model


def load_validation_data():
    dataset = XView2ChangeDataset(str(VAL_FILE))

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

                if hasattr(loader.dataset, "rows"):
                    scene_id = loader.dataset.rows[row_index]["scene_id"]
                else:
                    scene_id = "unknown"

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

    ax.set_title("Aegis AI — Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    ax.set_xticks(range(len(CLASS_NAMES)))
    ax.set_yticks(range(len(CLASS_NAMES)))

    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_yticklabels(CLASS_NAMES)

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                str(matrix[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)

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

        # Preserve multi-word disaster names.
        if scene_id.startswith("hurricane-"):
            disaster = "-".join(scene_id.split("-")[:2])

        grouped_labels[disaster].append(label)
        grouped_predictions[disaster].append(prediction)

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


def save_per_disaster_csv(results, path):
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
        writer.writerows(results)


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    device = get_device()

    print(f"Device: {device}")
    print(f"Validation file: {VAL_FILE}")
    print(f"Checkpoint: {CHECKPOINT}")

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VAL_FILE}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    print("\nLoading dataset...")
    dataset, loader = load_validation_data()

    print(f"Validation samples: {len(dataset)}")

    print("\nLoading model...")
    model = load_model(device)

    print("Running evaluation...")

    labels, predictions, scene_ids = evaluate(
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

    print("\nOverall evaluation")
    print("-" * 50)
    print(f"Accuracy:    {accuracy:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}")

    print("\nPer-class results")
    print("-" * 50)

    for class_name in CLASS_NAMES:
        metrics = report[class_name]

        print(
            f"{class_name:<15}"
            f" precision={metrics['precision']:.4f}"
            f" recall={metrics['recall']:.4f}"
            f" f1={metrics['f1-score']:.4f}"
        )

    print("\nConfusion matrix")
    print("-" * 50)
    print(matrix)

    print("\nPer-disaster evaluation")
    print("-" * 50)

    disaster_results = calculate_per_disaster(
        labels,
        predictions,
        scene_ids,
    )

    for result in disaster_results:
        print(
            f"{result['disaster']:<25}"
            f" samples={result['samples']:<6}"
            f" accuracy={result['accuracy']:.4f}"
            f" macro_f1={result['macro_f1']:.4f}"
        )

    summary = {
        "model": "AegisResNet18ChangeAware",
        "checkpoint": str(
            CHECKPOINT.relative_to(PROJECT_ROOT)
        ),
        "validation_samples": len(dataset),
        "device": str(device),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classes": CLASS_NAMES,
    }

    save_json(
        summary,
        OUTPUT_DIR / "summary.json",
    )

    save_json(
        report,
        OUTPUT_DIR / "classification_report.json",
    )

    save_json(
        matrix.tolist(),
        OUTPUT_DIR / "confusion_matrix.json",
    )

    save_confusion_matrix(
        matrix,
        OUTPUT_DIR / "confusion_matrix.png",
    )

    save_per_disaster_csv(
        disaster_results,
        OUTPUT_DIR / "per_disaster_f1.csv",
    )

    print("\nEvaluation files created:")
    print(
        f"  {OUTPUT_DIR / 'summary.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'classification_report.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'confusion_matrix.json'}"
    )
    print(
        f"  {OUTPUT_DIR / 'confusion_matrix.png'}"
    )
    print(
        f"  {OUTPUT_DIR / 'per_disaster_f1.csv'}"
    )


if __name__ == "__main__":
    main()