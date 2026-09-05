from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.dataset import XView2BuildingDataset
from src.models.baseline import AegisBaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"

BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 1e-3

NUM_CLASSES = 4

CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_dataloaders():
    train_dataset = XView2BuildingDataset(TRAIN_FILE)
    val_dataset = XView2BuildingDataset(VAL_FILE)

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return train_loader, val_loader


def calculate_class_weights(dataset):
    counts = torch.zeros(NUM_CLASSES)

    for row in dataset.rows:
        label = int(row["damage_class"])
        counts[label] += 1

    weights = counts.sum() / (NUM_CLASSES * counts)

    return weights


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    total = 0

    confusion_matrix = torch.zeros(
        NUM_CLASSES,
        NUM_CLASSES,
        dtype=torch.long,
    )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)
        total += labels.size(0)

        predictions = outputs.argmax(dim=1)

        for true_label, predicted_label in zip(
            labels.cpu(),
            predictions.cpu(),
        ):
            confusion_matrix[true_label, predicted_label] += 1

    epoch_loss = running_loss / total

    return epoch_loss, confusion_matrix


def calculate_metrics(confusion_matrix):
    metrics = {}

    total = confusion_matrix.sum().item()

    correct = torch.trace(confusion_matrix).item()

    accuracy = correct / total

    f1_scores = []

    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = confusion_matrix[
            class_index,
            class_index,
        ].item()

        false_positive = (
            confusion_matrix[:, class_index].sum().item()
            - true_positive
        )

        false_negative = (
            confusion_matrix[class_index, :].sum().item()
            - true_positive
        )

        precision_denominator = true_positive + false_positive
        recall_denominator = true_positive + false_negative

        precision = (
            true_positive / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = precision + recall

        f1 = (
            2 * precision * recall / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        metrics[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        f1_scores.append(f1)

    macro_f1 = sum(f1_scores) / NUM_CLASSES

    return accuracy, macro_f1, metrics


def print_metrics(
    accuracy,
    macro_f1,
    metrics,
):
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro F1: {macro_f1:.4f}")

    print("\nPer-class metrics:")

    for class_name in CLASS_NAMES:
        values = metrics[class_name]

        print(
            f"  {class_name:<15} "
            f"Precision: {values['precision']:.4f} "
            f"Recall: {values['recall']:.4f} "
            f"F1: {values['f1']:.4f}"
        )


def print_confusion_matrix(confusion_matrix):
    print("\nConfusion Matrix:")
    print(
        "Rows = true labels | Columns = predicted labels"
    )

    print(
        f"{'':>15}"
        + "".join(
            f"{name:>15}"
            for name in CLASS_NAMES
        )
    )

    for index, class_name in enumerate(CLASS_NAMES):
        values = confusion_matrix[index].tolist()

        print(
            f"{class_name:>15}"
            + "".join(
                f"{value:>15}"
                for value in values
            )
        )


def main():
    print("Aegis AI - Baseline Training")
    print("=" * 50)

    device = get_device()

    print(f"Device: {device}")

    train_loader, val_loader = create_dataloaders()

    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    class_weights = calculate_class_weights(
        train_loader.dataset
    )

    print("\nClass weights:")

    for class_name, weight in zip(
        CLASS_NAMES,
        class_weights,
    ):
        print(f"  {class_name:<15} {weight:.4f}")

    model = AegisBaselineCNN(
        num_classes=NUM_CLASSES
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_macro_f1 = -1.0

    for epoch in range(1, EPOCHS + 1):
        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )
        print("-" * 50)

        train_loss, train_accuracy = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        val_loss, confusion_matrix = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        val_accuracy, macro_f1, metrics = calculate_metrics(
            confusion_matrix
        )

        print(f"Train Loss: {train_loss:.4f}")
        print(f"Train Accuracy: {train_accuracy:.4f}")
        print(f"Validation Loss: {val_loss:.4f}")

        print_metrics(
            val_accuracy,
            macro_f1,
            metrics,
        )

        print_confusion_matrix(
            confusion_matrix
        )

        if macro_f1 > best_macro_f1:
            best_macro_f1 = macro_f1

            checkpoint_path = (
                CHECKPOINT_DIR / "best_baseline.pt"
            )

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "macro_f1": macro_f1,
                    "accuracy": val_accuracy,
                    "class_names": CLASS_NAMES,
                },
                checkpoint_path,
            )

            print(
                f"\nBest checkpoint saved: "
                f"{checkpoint_path}"
            )

    print("\nTraining complete.")
    print(f"Best Validation Macro F1: {best_macro_f1:.4f}")


if __name__ == "__main__":
    main()