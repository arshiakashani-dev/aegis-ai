from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from src.data.dataset import XView2BuildingDataset
from src.models.baseline import AegisBaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_baseline.pt"

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


def disaster_from_scene(scene_id):
    return scene_id.rsplit("_", 1)[0]


def confusion_matrix(y_true, y_pred, num_classes=4):
    matrix = torch.zeros(
        (num_classes, num_classes),
        dtype=torch.long,
    )

    for true, pred in zip(y_true, y_pred):
        matrix[true, pred] += 1

    return matrix


def metrics_from_confusion(cm):
    metrics = []

    for i in range(len(CLASS_NAMES)):
        tp = cm[i, i].item()
        fp = cm[:, i].sum().item() - tp
        fn = cm[i, :].sum().item() - tp

        precision = (
            tp / (tp + fp)
            if tp + fp
            else 0.0
        )

        recall = (
            tp / (tp + fn)
            if tp + fn
            else 0.0
        )

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )

        metrics.append(
            (precision, recall, f1)
        )

    macro_f1 = sum(
        metric[2] for metric in metrics
    ) / len(metrics)

    return metrics, macro_f1


def load_model(device):
    model = AegisBaselineCNN(
        num_classes=4
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
        weights_only=False,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model.eval()

    return model


def main():
    device = get_device()

    print()
    print("Aegis AI - Error Analysis")
    print("=" * 70)
    print("Device:", device)

    dataset = XView2BuildingDataset(
        VAL_FILE
    )

    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    model = load_model(device)

    all_true = []
    all_pred = []

    disaster_true = defaultdict(list)
    disaster_pred = defaultdict(list)

    offset = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            predictions = (
                outputs
                .argmax(dim=1)
                .cpu()
            )

            labels = labels.cpu()

            batch_size = len(labels)

            for i in range(batch_size):
                row = dataset.rows[
                    offset + i
                ]

                disaster = disaster_from_scene(
                    row["scene_id"]
                )

                true_label = labels[i].item()
                pred_label = predictions[i].item()

                all_true.append(true_label)
                all_pred.append(pred_label)

                disaster_true[
                    disaster
                ].append(true_label)

                disaster_pred[
                    disaster
                ].append(pred_label)

            offset += batch_size

    # --------------------------------------------------
    # DISASTER CLASS DISTRIBUTION
    # --------------------------------------------------

    print()
    print("DISASTER CLASS DISTRIBUTION")
    print("=" * 95)

    print(
        f"{'Disaster':28s}"
        f"{'no-damage':>14s}"
        f"{'minor':>14s}"
        f"{'major':>14s}"
        f"{'destroyed':>14s}"
        f"{'Total':>12s}"
    )

    print("-" * 95)

    for disaster in sorted(disaster_true):
        counts = [0, 0, 0, 0]

        for label in disaster_true[disaster]:
            counts[label] += 1

        total = sum(counts)

        print(
            f"{disaster:28s}"
            f"{counts[0]:14d}"
            f"{counts[1]:14d}"
            f"{counts[2]:14d}"
            f"{counts[3]:14d}"
            f"{total:12d}"
        )

    # --------------------------------------------------
    # PER-DISASTER RESULTS
    # --------------------------------------------------

    print()
    print("PER-DISASTER RESULTS")
    print("=" * 70)

    results = []

    for disaster in sorted(disaster_true):
        cm = confusion_matrix(
            disaster_true[disaster],
            disaster_pred[disaster],
        )

        class_metrics, macro_f1 = (
            metrics_from_confusion(cm)
        )

        correct = cm.diag().sum().item()
        total = cm.sum().item()

        accuracy = (
            correct / total
            if total
            else 0.0
        )

        results.append(
            (
                macro_f1,
                disaster,
                accuracy,
                total,
                class_metrics,
            )
        )

    results.sort()

    print(
        f"{'Disaster':28s}"
        f"{'Samples':>8s}"
        f"{'Accuracy':>10s}"
        f"{'Macro F1':>10s}"
    )

    print("-" * 70)

    for (
        macro_f1,
        disaster,
        accuracy,
        total,
        _,
    ) in results:
        print(
            f"{disaster:28s}"
            f"{total:8d}"
            f"{accuracy:10.4f}"
            f"{macro_f1:10.4f}"
        )

    # --------------------------------------------------
    # PER-CLASS F1 BY DISASTER
    # --------------------------------------------------

    print()
    print("PER-CLASS F1 BY DISASTER")
    print("=" * 95)

    print(
        f"{'Disaster':28s}"
        f"{'no-damage':>14s}"
        f"{'minor':>14s}"
        f"{'major':>14s}"
        f"{'destroyed':>14s}"
    )

    print("-" * 95)

    for (
        macro_f1,
        disaster,
        accuracy,
        total,
        class_metrics,
    ) in results:

        f1s = [
            metric[2]
            for metric in class_metrics
        ]

        print(
            f"{disaster:28s}"
            f"{f1s[0]:14.4f}"
            f"{f1s[1]:14.4f}"
            f"{f1s[2]:14.4f}"
            f"{f1s[3]:14.4f}"
        )

    # --------------------------------------------------
    # OVERALL RESULTS
    # --------------------------------------------------

    overall_cm = confusion_matrix(
        all_true,
        all_pred,
    )

    overall_metrics, overall_macro_f1 = (
        metrics_from_confusion(
            overall_cm
        )
    )

    print()
    print("OVERALL")
    print("=" * 70)

    print(
        "Macro F1:",
        f"{overall_macro_f1:.4f}",
    )

    print()
    print("Confusion Matrix:")

    print(
        f"{'':18s}"
        f"{'no-damage':>12s}"
        f"{'minor':>12s}"
        f"{'major':>12s}"
        f"{'destroyed':>12s}"
    )

    for i, name in enumerate(CLASS_NAMES):
        values = overall_cm[i].tolist()

        print(
            f"{name:18s}"
            f"{values[0]:12d}"
            f"{values[1]:12d}"
            f"{values[2]:12d}"
            f"{values[3]:12d}"
        )

    # --------------------------------------------------
    # LARGEST CONFUSIONS
    # --------------------------------------------------

    print()
    print("Largest Confusions:")
    print("=" * 70)

    confusions = []

    for true_idx in range(4):
        for pred_idx in range(4):

            if true_idx == pred_idx:
                continue

            count = overall_cm[
                true_idx,
                pred_idx,
            ].item()

            confusions.append(
                (
                    count,
                    CLASS_NAMES[true_idx],
                    CLASS_NAMES[pred_idx],
                )
            )

    confusions.sort(
        reverse=True
    )

    for (
        count,
        true_name,
        pred_name,
    ) in confusions[:10]:

        print(
            f"{true_name:15s}"
            f" -> "
            f"{pred_name:15s}"
            f": {count}"
        )

    # --------------------------------------------------
    # WORST / BEST DISASTER
    # --------------------------------------------------

    worst = results[0]
    best = results[-1]

    print()
    print("SUMMARY")
    print("=" * 70)

    print(
        f"Worst disaster: {worst[1]} "
        f"(Macro F1 = {worst[0]:.4f})"
    )

    print(
        f"Best disaster : {best[1]} "
        f"(Macro F1 = {best[0]:.4f})"
    )


if __name__ == "__main__":
    main()