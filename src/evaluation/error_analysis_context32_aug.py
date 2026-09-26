import csv
from collections import Counter, defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.context32_change_dataset import XView2Context32Dataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "val.csv"
)

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

ERROR_CSV = OUTPUT_DIR / "error_analysis.csv"
CONFUSION_CSV = OUTPUT_DIR / "error_confusion_pairs.csv"
DISASTER_CSV = OUTPUT_DIR / "error_by_disaster.csv"

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


def evaluate_errors(model, loader, device):
    errors = []

    sample_index = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            predictions = outputs.argmax(
                dim=1
            )

            confidence = probabilities.max(
                dim=1
            ).values

            batch_size = len(labels)

            for i in range(batch_size):
                true_label = int(
                    labels[i].item()
                )

                predicted_label = int(
                    predictions[i].item()
                )

                if true_label == predicted_label:
                    continue

                row_index = sample_index + i

                scene_id = (
                    loader.dataset.rows[row_index][
                        "scene_id"
                    ]
                )

                disaster = get_disaster_name(
                    scene_id
                )

                errors.append(
                    {
                        "sample_index": row_index,
                        "scene_id": scene_id,
                        "disaster": disaster,
                        "true_label": true_label,
                        "true_class": CLASS_NAMES[
                            true_label
                        ],
                        "predicted_label": predicted_label,
                        "predicted_class": CLASS_NAMES[
                            predicted_label
                        ],
                        "confidence": float(
                            confidence[i].item()
                        ),
                    }
                )

            sample_index += batch_size

    return errors


def save_error_csv(errors):
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "sample_index",
        "scene_id",
        "disaster",
        "true_label",
        "true_class",
        "predicted_label",
        "predicted_class",
        "confidence",
    ]

    with open(
        ERROR_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(errors)


def analyze_confusion_pairs(errors):
    pairs = Counter()

    for error in errors:
        pair = (
            error["true_class"],
            error["predicted_class"],
        )

        pairs[pair] += 1

    rows = []

    for (true_class, predicted_class), count in (
        pairs.most_common()
    ):
        rows.append(
            {
                "true_class": true_class,
                "predicted_class": predicted_class,
                "errors": count,
            }
        )

    with open(
        CONFUSION_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "true_class",
                "predicted_class",
                "errors",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def analyze_disasters(errors):
    grouped = defaultdict(list)

    for error in errors:
        grouped[
            error["disaster"]
        ].append(error)

    rows = []

    for disaster in sorted(grouped):
        disaster_errors = grouped[disaster]

        pair_counter = Counter(
            (
                error["true_class"],
                error["predicted_class"],
            )
            for error in disaster_errors
        )

        most_common_pair = (
            pair_counter.most_common(1)[0]
            if pair_counter
            else (("unknown", "unknown"), 0)
        )

        rows.append(
            {
                "disaster": disaster,
                "errors": len(disaster_errors),
                "top_error_true": (
                    most_common_pair[0][0]
                ),
                "top_error_predicted": (
                    most_common_pair[0][1]
                ),
                "top_error_count": (
                    most_common_pair[1]
                ),
            }
        )

    rows.sort(
        key=lambda row: row["errors"],
        reverse=True,
    )

    with open(
        DISASTER_CSV,
        "w",
        newline="",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "disaster",
                "errors",
                "top_error_true",
                "top_error_predicted",
                "top_error_count",
            ],
        )

        writer.writeheader()
        writer.writerows(rows)

    return rows


def print_report(
    errors,
    confusion_pairs,
    disaster_rows,
):
    print()
    print(
        "Aegis AI - Context32 + Augmentation Error Analysis"
    )
    print(
        "=" * 60
    )

    print()
    print(
        f"Total misclassified samples: {len(errors)}"
    )

    print()
    print(
        "Top confusion pairs"
    )
    print(
        "-" * 60
    )

    for row in confusion_pairs[:10]:
        print(
            f"{row['true_class']:>15} -> "
            f"{row['predicted_class']:<15} "
            f"errors={row['errors']}"
        )

    print()
    print(
        "Most affected disasters"
    )
    print(
        "-" * 60
    )

    for row in disaster_rows[:15]:
        print(
            f"{row['disaster']:<25} "
            f"errors={row['errors']:<6} "
            f"top={row['top_error_true']} -> "
            f"{row['top_error_predicted']} "
            f"({row['top_error_count']})"
        )

    print()
    print(
        "Important targeted errors"
    )
    print(
        "-" * 60
    )

    targeted_pairs = [
        (
            "minor-damage",
            "major-damage",
        ),
        (
            "major-damage",
            "minor-damage",
        ),
        (
            "destroyed",
            "major-damage",
        ),
        (
            "major-damage",
            "destroyed",
        ),
        (
            "no-damage",
            "minor-damage",
        ),
    ]

    pair_counts = Counter(
        (
            error["true_class"],
            error["predicted_class"],
        )
        for error in errors
    )

    for true_class, predicted_class in targeted_pairs:
        count = pair_counts[
            (true_class, predicted_class)
        ]

        print(
            f"{true_class:>15} -> "
            f"{predicted_class:<15} "
            f"errors={count}"
        )

    print()
    print(
        "Outputs:"
    )
    print(
        ERROR_CSV
    )
    print(
        CONFUSION_CSV
    )
    print(
        DISASTER_CSV
    )

    print()


def main():
    device = get_device()

    print()
    print(
        "Aegis AI - Context32 + Augmentation Error Analysis"
    )
    print(
        "=" * 60
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

    _, loader = load_validation_data()

    errors = evaluate_errors(
        model,
        loader,
        device,
    )

    save_error_csv(errors)

    confusion_pairs = analyze_confusion_pairs(
        errors
    )

    disaster_rows = analyze_disasters(
        errors
    )

    print_report(
        errors,
        confusion_pairs,
        disaster_rows,
    )


if __name__ == "__main__":
    main()