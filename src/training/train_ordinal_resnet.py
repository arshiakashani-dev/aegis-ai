import os
import random

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.data.change_dataset import XView2ChangeDataset
from src.models.ordinal_resnet import (
    AegisOrdinalResNet18,
    ordinal_targets,
    ordinal_predictions,
)


SEED = 42
BATCH_SIZE = 16
LEARNING_RATE = 1e-3
EPOCHS = 5
NUM_WORKERS = 0

TRAIN_MANIFEST = "data/processed/train.csv"
VAL_MANIFEST = "data/processed/val.csv"
CHECKPOINT_PATH = "checkpoints/best_ordinal_resnet.pt"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def compute_metrics(confusion):
    num_classes = confusion.shape[0]

    precision = []
    recall = []
    f1 = []

    for class_id in range(num_classes):
        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp

        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        if p + r > 0:
            score = 2 * p * r / (p + r)
        else:
            score = 0.0

        precision.append(p)
        recall.append(r)
        f1.append(score)

    accuracy = np.trace(confusion) / confusion.sum()

    return accuracy, np.array(precision), np.array(recall), np.array(f1)


def evaluate(model, loader, device, criterion):
    model.eval()

    total_loss = 0.0
    total_samples = 0

    confusion = np.zeros((4, 4), dtype=np.int64)

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            targets = ordinal_targets(labels)

            loss = criterion(logits, targets)

            predictions = ordinal_predictions(logits)

            total_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            for true_label, predicted_label in zip(
                labels.cpu().numpy(),
                predictions.cpu().numpy(),
            ):
                confusion[true_label, predicted_label] += 1

    loss = total_loss / total_samples

    accuracy, precision, recall, f1 = compute_metrics(confusion)

    macro_f1 = f1.mean()

    return loss, accuracy, macro_f1, precision, recall, f1, confusion


def main():
    set_seed(SEED)

    print("Aegis AI - Ordinal ResNet18 Training")
    print("=" * 50)

    device = get_device()

    print(f"Device: {device}")

    train_dataset = XView2ChangeDataset(TRAIN_MANIFEST)
    val_dataset = XView2ChangeDataset(VAL_MANIFEST)

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

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Ordinal target distributions:
    #
    # threshold 0: damage >= minor
    # threshold 1: damage >= major
    # threshold 2: damage >= destroyed
    #
    # Positive counts in the training split:
    #   >= minor     = 42368
    #   >= major     = 27388
    #   >= destroyed = 13227
    #
    # pos_weight = negative / positive
    #
    # These weights compensate for the increasing rarity
    # of higher-severity thresholds.

    pos_weight = torch.tensor(
        [
            117426 / 42368,
            132406 / 27388,
            146567 / 13227,
        ],
        dtype=torch.float32,
        device=device,
    )

    print("Ordinal positive weights:")
    print(f"damage >= minor     {pos_weight[0].item():.4f}")
    print(f"damage >= major     {pos_weight[1].item():.4f}")
    print(f"damage >= destroyed {pos_weight[2].item():.4f}")

    model = AegisOrdinalResNet18(
        pretrained=True,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight,
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_macro_f1 = -1.0
    best_epoch = 0

    os.makedirs(
        os.path.dirname(CHECKPOINT_PATH),
        exist_ok=True,
    )

    for epoch in range(1, EPOCHS + 1):
        print()
        print(f"Epoch {epoch}")
        print("-" * 50)

        model.train()

        running_loss = 0.0
        total_samples = 0

        train_confusion = np.zeros(
            (4, 4),
            dtype=np.int64,
        )

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            logits = model(images)

            targets = ordinal_targets(labels)

            loss = criterion(
                logits,
                targets,
            )

            loss.backward()

            optimizer.step()

            predictions = ordinal_predictions(logits)

            running_loss += loss.item() * labels.size(0)
            total_samples += labels.size(0)

            for true_label, predicted_label in zip(
                labels.detach().cpu().numpy(),
                predictions.detach().cpu().numpy(),
            ):
                train_confusion[
                    true_label,
                    predicted_label,
                ] += 1

        train_loss = running_loss / total_samples

        (
            train_accuracy,
            _,
            _,
            train_f1,
        ) = compute_metrics(train_confusion)

        train_macro_f1 = train_f1.mean()

        (
            val_loss,
            val_accuracy,
            val_macro_f1,
            val_precision,
            val_recall,
            val_f1,
            val_confusion,
        ) = evaluate(
            model,
            val_loader,
            device,
            criterion,
        )

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

        print()
        print("Validation per-class metrics:")

        class_names = [
            "no-damage",
            "minor-damage",
            "major-damage",
            "destroyed",
        ]

        for class_id, class_name in enumerate(class_names):
            print(
                f"{class_name:<12} "
                f"P={val_precision[class_id]:.4f} "
                f"R={val_recall[class_id]:.4f} "
                f"F1={val_f1[class_id]:.4f}"
            )

        print()
        print("Validation confusion matrix:")
        print(val_confusion)

        if val_macro_f1 > best_macro_f1:
            best_macro_f1 = val_macro_f1
            best_epoch = epoch

            torch.save(
                {
                    "epoch": epoch,
                    "macro_f1": val_macro_f1,
                    "model_state_dict": model.state_dict(),
                },
                CHECKPOINT_PATH,
            )

            print()
            print(
                f"New best checkpoint saved: "
                f"{os.path.abspath(CHECKPOINT_PATH)}"
            )

    print()
    print("=" * 50)
    print("Training complete")
    print()
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation Macro F1: {best_macro_f1:.4f}")
    print(f"Checkpoint: {os.path.abspath(CHECKPOINT_PATH)}")


if __name__ == "__main__":
    main()
