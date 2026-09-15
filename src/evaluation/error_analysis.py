import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.change_dataset import XView2ChangeDataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_resnet_change_aware.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis"

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


def get_disaster(scene_id):
    return scene_id.rsplit("_", 1)[0]


def run_predictions(model, dataset, loader, device):
    predictions = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            probabilities = torch.softmax(outputs, dim=1)

            confidences, predicted_classes = probabilities.max(dim=1)

            batch_size = len(labels)

            for i in range(batch_size):
                row = dataset.rows[sample_index + i]

                true_class = int(labels[i].item())
                predicted_class = int(predicted_classes[i].item())

                probability_values = probabilities[i].cpu().tolist()

                predictions.append(
                    {
                        "scene_id": row["scene_id"],
                        "uid": row["uid"],
                        "disaster": get_disaster(row["scene_id"]),
                        "true_class": true_class,
                        "true_label": CLASS_NAMES[true_class],
                        "predicted_class": predicted_class,
                        "predicted_label": CLASS_NAMES[predicted_class],
                        "correct": true_class == predicted_class,
                        "confidence": float(confidences[i].item()),
                        "prob_no_damage": float(probability_values[0]),
                        "prob_minor": float(probability_values[1]),
                        "prob_major": float(probability_values[2]),
                        "prob_destroyed": float(probability_values[3]),
                        "pre_image": row["pre_image"],
                        "post_image": row["post_image"],
                        "polygon_wkt": row["polygon_wkt"],
                    }
                )

            sample_index += batch_size

    return predictions


def save_predictions(predictions):
    path = OUTPUT_DIR / "predictions.csv"

    fieldnames = [
        "scene_id",
        "uid",
        "disaster",
        "true_class",
        "true_label",
        "predicted_class",
        "predicted_label",
        "correct",
        "confidence",
        "prob_no_damage",
        "prob_minor",
        "prob_major",
        "prob_destroyed",
        "pre_image",
        "post_image",
        "polygon_wkt",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(predictions)

    return path


def save_errors(predictions):
    errors = [
        row
        for row in predictions
        if not row["correct"]
    ]

    path = OUTPUT_DIR / "errors.csv"

    fieldnames = list(predictions[0].keys())

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(errors)

    return errors, path


def save_confusion_pairs(errors):
    counts = Counter(
        (
            row["true_label"],
            row["predicted_label"],
        )
        for row in errors
    )

    rows = []

    for (true_label, predicted_label), count in counts.most_common():
        rows.append(
            {
                "true_label": true_label,
                "predicted_label": predicted_label,
                "errors": count,
            }
        )

    path = OUTPUT_DIR / "confusion_pairs.csv"

    fieldnames = [
        "true_label",
        "predicted_label",
        "errors",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_high_confidence_errors(errors):
    rows = sorted(
        errors,
        key=lambda row: row["confidence"],
        reverse=True,
    )

    path = OUTPUT_DIR / "high_confidence_errors.csv"

    fieldnames = list(errors[0].keys()) if errors else [
        "scene_id",
        "uid",
        "disaster",
        "true_label",
        "predicted_label",
        "confidence",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def save_per_disaster_errors(predictions):
    grouped = defaultdict(list)

    for row in predictions:
        grouped[row["disaster"]].append(row)

    rows = []

    for disaster in sorted(grouped):
        disaster_rows = grouped[disaster]

        errors = [
            row
            for row in disaster_rows
            if not row["correct"]
        ]

        rows.append(
            {
                "disaster": disaster,
                "samples": len(disaster_rows),
                "errors": len(errors),
                "error_rate": len(errors) / len(disaster_rows),
            }
        )

    path = OUTPUT_DIR / "per_disaster_errors.csv"

    fieldnames = [
        "disaster",
        "samples",
        "errors",
        "error_rate",
    ]

    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    return path


def print_summary(predictions, errors):
    print("\nError Analysis")
    print("-" * 50)

    print(f"Total predictions: {len(predictions)}")
    print(f"Total errors:      {len(errors)}")
    print(
        f"Error rate:        "
        f"{len(errors) / len(predictions):.4f}"
    )

    print("\nMost common confusion pairs")
    print("-" * 50)

    counts = Counter(
        (
            row["true_label"],
            row["predicted_label"],
        )
        for row in errors
    )

    for (true_label, predicted_label), count in counts.most_common(10):
        print(
            f"{true_label:<15} -> "
            f"{predicted_label:<15} "
            f"{count}"
        )

    print("\nHighest-confidence errors")
    print("-" * 50)

    for row in sorted(
        errors,
        key=lambda row: row["confidence"],
        reverse=True,
    )[:10]:
        print(
            f"{row['true_label']:<15} -> "
            f"{row['predicted_label']:<15} "
            f"confidence={row['confidence']:.4f} "
            f"scene={row['scene_id']} "
            f"uid={row['uid']}"
        )


def main():
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not VAL_FILE.exists():
        raise FileNotFoundError(
            f"Validation file not found: {VAL_FILE}"
        )

    if not CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT}"
        )

    device = get_device()

    print(f"Device: {device}")
    print(f"Validation file: {VAL_FILE}")
    print(f"Checkpoint: {CHECKPOINT}")

    print("\nLoading validation data...")
    dataset, loader = load_validation_data()

    print(f"Validation samples: {len(dataset)}")

    print("\nLoading model...")
    model = load_model(device)

    print("Running predictions...")
    predictions = run_predictions(
        model,
        dataset,
        loader,
        device,
    )

    errors, errors_path = save_errors(predictions)

    predictions_path = save_predictions(predictions)
    confusion_path = save_confusion_pairs(errors)
    high_confidence_path = save_high_confidence_errors(errors)
    disaster_path = save_per_disaster_errors(predictions)

    print_summary(
        predictions,
        errors,
    )

    print("\nFiles created")
    print("-" * 50)
    print(predictions_path)
    print(errors_path)
    print(confusion_path)
    print(high_confidence_path)
    print(disaster_path)


if __name__ == "__main__":
    main()
