from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.change_dataset import XView2ChangeDataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_MANIFEST = PROJECT_ROOT / "data" / "processed" / "train.csv"
VAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "val.csv"

CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_resnet_change_aware.pt"


# Baseline training configuration
BATCH_SIZE = 16
NUM_WORKERS = 0
LEARNING_RATE = 1e-3
NUM_EPOCHS = 5

CLASS_WEIGHTS = torch.tensor(
    [0.3403, 2.6599, 2.9398, 2.8971],
    dtype=torch.float32,
)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def calculate_metrics(confusion):
    num_classes = confusion.shape[0]

    precision = []
    recall = []
    f1 = []

    for class_id in range(num_classes):
        true_positive = confusion[class_id, class_id]

        predicted = confusion[:, class_id].sum()
        actual = confusion[class_id, :].sum()

        p = (
            true_positive / predicted
            if predicted > 0
            else 0.0
        )

        r = (
            true_positive / actual
            if actual > 0
            else 0.0
        )

        score = (
            2 * p * r / (p + r)
            if p + r > 0
            else 0.0
        )

        precision.append(p)
        recall.append(r)
        f1.append(score)

    macro_f1 = sum(f1) / num_classes

    return precision, recall, f1, macro_f1


def run_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    training,
):
    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0
    total_samples = 0

    confusion = torch.zeros(
        4,
        4,
        dtype=torch.long,
    )

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if training:
            optimizer.zero_grad()

        with torch.set_grad_enabled(training):
            outputs = model(images)

            loss = criterion(
                outputs,
                labels,
            )

            if training:
                loss.backward()
                optimizer.step()

        total_loss += (
            loss.item() * labels.size(0)
        )

        total_samples += labels.size(0)

        predictions = outputs.argmax(dim=1)

        for true_label, predicted_label in zip(
            labels.detach().cpu(),
            predictions.detach().cpu(),
        ):
            confusion[
                true_label,
                predicted_label,
            ] += 1

    average_loss = (
        total_loss / total_samples
    )

    accuracy = (
        confusion.diag().sum().item()
        / total_samples
    )

    (
        precision,
        recall,
        f1,
        macro_f1,
    ) = calculate_metrics(confusion)

    return (
        average_loss,
        accuracy,
        macro_f1,
        precision,
        recall,
        f1,
        confusion,
    )


def print_metrics(
    epoch,
    train_metrics,
    val_metrics,
):
    (
        train_loss,
        train_accuracy,
        train_macro_f1,
        _,
        _,
        _,
        _,
    ) = train_metrics

    (
        val_loss,
        val_accuracy,
        val_macro_f1,
        precision,
        recall,
        f1,
        confusion,
    ) = val_metrics

    print(f"\nEpoch {epoch}")
    print("-" * 50)

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Accuracy: {train_accuracy:.4f} | "
        f"Train Macro F1: {train_macro_f1:.4f}"
    )

    print(
        f"Val Loss: {val_loss:.4f} | "
        f"Val Accuracy: {val_accuracy:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )

    class_names = [
        "no-damage",
        "minor-damage",
        "major-damage",
        "destroyed",
    ]

    print("\nValidation per-class metrics:")

    for index, name in enumerate(class_names):
        print(
            f"{name:12s} "
            f"P={precision[index]:.4f} "
            f"R={recall[index]:.4f} "
            f"F1={f1[index]:.4f}"
        )

    print("\nValidation confusion matrix:")
    print(confusion.numpy())


def main():
    print("Aegis AI - ResNet18 Change-Aware Training")
    print("=" * 50)

    print("Experiment: baseline reproduction")

    device = get_device()

    print("Device:", device)

    # ---------------------------------------------------------
    # Dataset
    # ---------------------------------------------------------

    train_dataset = XView2ChangeDataset(
        TRAIN_MANIFEST,
        augment=False,
    )

    val_dataset = XView2ChangeDataset(
        VAL_MANIFEST,
        augment=False,
    )

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(val_dataset),
    )

    print("Training augmentation: OFF")
    print("Validation augmentation: OFF")

    # ---------------------------------------------------------
    # DataLoaders
    # ---------------------------------------------------------

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

    # ---------------------------------------------------------
    # Class weights
    # ---------------------------------------------------------

    class_weights = CLASS_WEIGHTS.to(device)

    print("\nClass weights:")

    class_names = [
        "no-damage",
        "minor-damage",
        "major-damage",
        "destroyed",
    ]

    for name, weight in zip(
        class_names,
        CLASS_WEIGHTS,
    ):
        print(
            f"  {name:12s} "
            f"{weight.item():.4f}"
        )

    # ---------------------------------------------------------
    # Model
    # ---------------------------------------------------------

    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=True,
    ).to(device)

    # ---------------------------------------------------------
    # Loss + optimizer
    # ---------------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    # ---------------------------------------------------------
    # Training
    # ---------------------------------------------------------

    best_macro_f1 = -1.0
    best_epoch = -1

    for epoch in range(
        1,
        NUM_EPOCHS + 1,
    ):
        train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            training=True,
        )

        val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            optimizer=None,
            device=device,
            training=False,
        )

        print_metrics(
            epoch,
            train_metrics,
            val_metrics,
        )

        val_macro_f1 = val_metrics[2]

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_macro_f1": val_macro_f1,
                },
                CHECKPOINT_PATH,
            )

            print(
                f"\nNew best checkpoint saved: "
                f"{CHECKPOINT_PATH}"
            )

    # ---------------------------------------------------------
    # Final result
    # ---------------------------------------------------------

    print("\n" + "=" * 50)
    print("Training complete")

    print(
        "Best epoch:",
        best_epoch,
    )

    print(
        f"Best validation Macro F1: "
        f"{best_macro_f1:.4f}"
    )

    print(
        "Checkpoint:",
        CHECKPOINT_PATH,
    )


if __name__ == "__main__":
    main()