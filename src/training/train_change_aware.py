import csv
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.change_dataset import XView2ChangeDataset
from src.models.change_aware import AegisChangeAwareCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_FILE = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "best_change_aware.pt"

BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
EPOCHS = 5


CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]


CLASS_WEIGHTS = torch.tensor(
    [
        0.3403,
        2.6599,
        2.9398,
        2.8971,
    ],
    dtype=torch.float32,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def build_confusion_matrix(
    predictions,
    labels,
    num_classes=4,
):
    matrix = torch.zeros(
        num_classes,
        num_classes,
        dtype=torch.long,
    )

    for true, pred in zip(
        labels,
        predictions,
    ):
        matrix[true, pred] += 1

    return matrix


def calculate_metrics(confusion_matrix):
    metrics = []

    for class_index in range(
        len(CLASS_NAMES)
    ):
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

        precision_denominator = (
            true_positive
            + false_positive
        )

        recall_denominator = (
            true_positive
            + false_negative
        )

        precision = (
            true_positive
            / precision_denominator
            if precision_denominator > 0
            else 0.0
        )

        recall = (
            true_positive
            / recall_denominator
            if recall_denominator > 0
            else 0.0
        )

        f1_denominator = (
            precision
            + recall
        )

        f1 = (
            2 * precision * recall
            / f1_denominator
            if f1_denominator > 0
            else 0.0
        )

        metrics.append(
            {
                "precision": precision,
                "recall": recall,
                "f1": f1,
            }
        )

    macro_f1 = sum(
        metric["f1"]
        for metric in metrics
    ) / len(metrics)

    total = confusion_matrix.sum().item()

    correct = sum(
        confusion_matrix[i, i].item()
        for i in range(
            len(CLASS_NAMES)
        )
    )

    accuracy = (
        correct / total
        if total > 0
        else 0.0
    )

    return (
        accuracy,
        macro_f1,
        metrics,
    )


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
):
    model.train()

    running_loss = 0.0
    predictions = []
    labels = []

    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            batch_labels,
        )

        loss.backward()
        optimizer.step()

        running_loss += (
            loss.item()
            * images.size(0)
        )

        batch_predictions = (
            outputs.argmax(dim=1)
        )

        predictions.extend(
            batch_predictions.detach()
            .cpu()
            .tolist()
        )

        labels.extend(
            batch_labels.detach()
            .cpu()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    confusion_matrix = (
        build_confusion_matrix(
            predictions,
            labels,
        )
    )

    accuracy, macro_f1, metrics = (
        calculate_metrics(
            confusion_matrix
        )
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        confusion_matrix,
        metrics,
    )


@torch.no_grad()
def evaluate(
    model,
    loader,
    criterion,
    device,
):
    model.eval()

    running_loss = 0.0
    predictions = []
    labels = []

    for images, batch_labels in loader:
        images = images.to(device)
        batch_labels = batch_labels.to(device)

        outputs = model(images)

        loss = criterion(
            outputs,
            batch_labels,
        )

        running_loss += (
            loss.item()
            * images.size(0)
        )

        batch_predictions = (
            outputs.argmax(dim=1)
        )

        predictions.extend(
            batch_predictions
            .cpu()
            .tolist()
        )

        labels.extend(
            batch_labels
            .cpu()
            .tolist()
        )

    epoch_loss = (
        running_loss
        / len(loader.dataset)
    )

    confusion_matrix = (
        build_confusion_matrix(
            predictions,
            labels,
        )
    )

    accuracy, macro_f1, metrics = (
        calculate_metrics(
            confusion_matrix
        )
    )

    return (
        epoch_loss,
        accuracy,
        macro_f1,
        confusion_matrix,
        metrics,
    )


def print_metrics(
    split_name,
    loss,
    accuracy,
    macro_f1,
    metrics,
):
    print(
        f"{split_name} Loss: "
        f"{loss:.4f}"
    )

    print(
        f"{split_name} Accuracy: "
        f"{accuracy:.4f}"
    )

    print(
        f"{split_name} Macro F1: "
        f"{macro_f1:.4f}"
    )

    for class_name, metric in zip(
        CLASS_NAMES,
        metrics,
    ):
        print(
            f"  {class_name:<14} "
            f"P={metric['precision']:.4f} "
            f"R={metric['recall']:.4f} "
            f"F1={metric['f1']:.4f}"
        )


def save_checkpoint(
    model,
    optimizer,
    epoch,
    macro_f1,
):
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "macro_f1": macro_f1,
        },
        CHECKPOINT_FILE,
    )


def main():
    print()
    print(
        "Aegis AI - Change-Aware Training"
    )
    print("=" * 50)

    device = get_device()

    print("Device:", device)

    print()
    print(
        "Training samples:",
        len(
            XView2ChangeDataset(
                TRAIN_FILE
            )
        ),
    )

    print(
        "Validation samples:",
        len(
            XView2ChangeDataset(
                VAL_FILE
            )
        ),
    )

    train_dataset = (
        XView2ChangeDataset(
            TRAIN_FILE
        )
    )

    val_dataset = (
        XView2ChangeDataset(
            VAL_FILE
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    model = AegisChangeAwareCNN(
        num_classes=4
    ).to(device)

    criterion = nn.CrossEntropyLoss(
        weight=CLASS_WEIGHTS.to(device)
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_epoch = -1

    print()
    print("Starting training...")
    print()

    for epoch in range(
        1,
        EPOCHS + 1,
    ):
        (
            train_loss,
            train_accuracy,
            train_macro_f1,
            _,
            train_metrics,
        ) = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
        )

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_confusion,
            val_metrics,
        ) = evaluate(
            model,
            val_loader,
            criterion,
            device,
        )

        print(
            f"Epoch {epoch}/{EPOCHS}"
        )
        print("-" * 50)

        print_metrics(
            "Train",
            train_loss,
            train_accuracy,
            train_macro_f1,
            train_metrics,
        )

        print()

        print_metrics(
            "Val",
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_metrics,
        )

        print()
        print(
            "Validation confusion matrix:"
        )
        print(val_confusion)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            save_checkpoint(
                model,
                optimizer,
                epoch,
                val_macro_f1,
            )

            print()
            print(
                f"New best model saved "
                f"(Macro F1: "
                f"{val_macro_f1:.4f})"
            )

        print()
        print("=" * 50)
        print()

    print(
        "Training complete."
    )

    print(
        f"Best epoch: {best_epoch}"
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        f"Checkpoint: "
        f"{CHECKPOINT_FILE}"
    )


if __name__ == "__main__":
    main()