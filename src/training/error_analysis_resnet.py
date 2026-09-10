from pathlib import Path
import csv
from collections import defaultdict

import torch
import numpy as np

from src.data.change_dataset import XView2ChangeDataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_resnet_change_aware.pt"


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


def calculate_metrics(confusion):
    num_classes = confusion.shape[0]

    precision = []
    recall = []
    f1 = []

    for class_id in range(num_classes):
        tp = confusion[class_id, class_id]

        predicted = confusion[:, class_id].sum()
        actual = confusion[class_id, :].sum()

        p = tp / predicted if predicted > 0 else 0.0
        r = tp / actual if actual > 0 else 0.0

        if p + r > 0:
            score = 2 * p * r / (p + r)
        else:
            score = 0.0

        precision.append(p)
        recall.append(r)
        f1.append(score)

    macro_f1 = sum(f1) / num_classes

    return precision, recall, f1, macro_f1


def disaster_from_scene(scene_id):
    return scene_id.rsplit("_", 1)[0]


def main():
    print("Aegis AI - ResNet18 Error Analysis")
    print("=" * 50)

    device = get_device()
    print("Device:", device)

    dataset = XView2ChangeDataset(VAL_MANIFEST)

    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    print("Loaded checkpoint:", CHECKPOINT)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print(
        f"Checkpoint Macro F1: "
        f"{checkpoint['val_macro_f1']:.4f}"
    )

    overall_confusion = np.zeros(
        (4, 4),
        dtype=np.int64,
    )

    disaster_confusions = defaultdict(
        lambda: np.zeros((4, 4), dtype=np.int64)
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
    )

    with torch.no_grad():
        offset = 0

        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu().numpy()
            labels = labels.numpy()

            for true_label, predicted_label in zip(
                labels,
                predictions,
            ):
                overall_confusion[
                    true_label,
                    predicted_label,
                ] += 1

                scene_id = dataset.rows[offset]["scene_id"]
                disaster = disaster_from_scene(scene_id)

                disaster_confusions[disaster][
                    true_label,
                    predicted_label,
                ] += 1

                offset += 1

    precision, recall, f1, macro_f1 = calculate_metrics(
        overall_confusion
    )

    total = overall_confusion.sum()
    accuracy = np.trace(overall_confusion) / total

    print("\nOverall performance")
    print("-" * 50)
    print(f"Accuracy : {accuracy:.4f}")
    print(f"Macro F1 : {macro_f1:.4f}")

    print("\nPer-class metrics")

    for index, name in enumerate(CLASS_NAMES):
        print(
            f"{name:12s} "
            f"P={precision[index]:.4f} "
            f"R={recall[index]:.4f} "
            f"F1={f1[index]:.4f}"
        )

    print("\nOverall confusion matrix")
    print(overall_confusion)

    print("\nLargest confusions")
    print("-" * 50)

    confusions = []

    for true_label in range(4):
        for predicted_label in range(4):
            if true_label == predicted_label:
                continue

            count = overall_confusion[
                true_label,
                predicted_label,
            ]

            confusions.append(
                (
                    count,
                    CLASS_NAMES[true_label],
                    CLASS_NAMES[predicted_label],
                )
            )

    confusions.sort(reverse=True)

    for count, true_name, predicted_name in confusions[:10]:
        print(
            f"{true_name:12s} -> "
            f"{predicted_name:12s}: {count}"
        )

    print("\nPerformance by disaster")
    print("-" * 50)

    disaster_results = []

    for disaster, confusion in disaster_confusions.items():
        disaster_total = confusion.sum()

        disaster_accuracy = (
            np.trace(confusion) / disaster_total
        )

        _, _, _, disaster_macro_f1 = calculate_metrics(
            confusion
        )

        disaster_results.append(
            (
                disaster_macro_f1,
                disaster_accuracy,
                disaster,
                disaster_total,
            )
        )

    disaster_results.sort()

    print(
        f"{'Disaster':24s} "
        f"{'Accuracy':>10s} "
        f"{'Macro F1':>10s} "
        f"{'Samples':>10s}"
    )

    for macro, acc, disaster, count in disaster_results:
        print(
            f"{disaster:24s} "
            f"{acc:10.4f} "
            f"{macro:10.4f} "
            f"{count:10d}"
        )

    print("\nPer-class F1 by disaster")
    print("-" * 50)

    for _, _, disaster, _ in disaster_results:
        confusion = disaster_confusions[disaster]

        _, _, disaster_f1, _ = calculate_metrics(
            confusion
        )

        print(f"\n{disaster}")

        for index, name in enumerate(CLASS_NAMES):
            print(
                f"  {name:12s}: "
                f"{disaster_f1[index]:.4f}"
            )


if __name__ == "__main__":
    main()
