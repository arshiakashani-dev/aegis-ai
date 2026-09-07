from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torch.utils.data import DataLoader

from src.data.dataset import XView2BuildingDataset
from src.models.baseline import AegisBaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_FILE = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_baseline.pt"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis"

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


def collect_predictions(dataset, model, device):
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=0,
    )

    predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)

            preds = (
                outputs
                .argmax(dim=1)
                .cpu()
            )

            predictions.extend(
                preds.tolist()
            )

    return predictions


def select_examples(
    dataset,
    predictions,
    true_label,
    pred_label,
    limit=12,
):
    examples = []

    for index, prediction in enumerate(
        predictions
    ):
        actual_label = CLASS_NAMES.index(
            dataset.rows[index]["damage_label"]
        )

        if (
            actual_label == true_label
            and prediction == pred_label
        ):
            examples.append(index)

        if len(examples) >= limit:
            break

    return examples


def create_grid(
    dataset,
    predictions,
    indices,
    true_label,
    pred_label,
    output_path,
):
    if not indices:
        print(
            f"No examples found: "
            f"{CLASS_NAMES[true_label]} -> "
            f"{CLASS_NAMES[pred_label]}"
        )
        return

    fig, axes = plt.subplots(
        len(indices),
        2,
        figsize=(8, 4 * len(indices)),
    )

    if len(indices) == 1:
        axes = [axes]

    for row_number, index in enumerate(
        indices
    ):
        row = dataset.rows[index]

        scene_id = row["scene_id"]
        uid = row["uid"]

        sample_dir = (
            dataset.cache_dir
            / scene_id
        )

        pre_path = (
            sample_dir
            / f"{uid}_pre.jpg"
        )

        post_path = (
            sample_dir
            / f"{uid}_post.jpg"
        )

        with Image.open(pre_path) as image:
            pre_image = image.convert("RGB")

        with Image.open(post_path) as image:
            post_image = image.convert("RGB")

        axes[row_number][0].imshow(
            pre_image
        )

        axes[row_number][0].set_title(
            f"PRE\nscene={scene_id}"
        )

        axes[row_number][0].axis("off")

        axes[row_number][1].imshow(
            post_image
        )

        axes[row_number][1].set_title(
            f"POST\n"
            f"True: {CLASS_NAMES[true_label]} | "
            f"Pred: {CLASS_NAMES[pred_label]}"
        )

        axes[row_number][1].axis("off")

    fig.suptitle(
        f"{CLASS_NAMES[true_label]} → "
        f"{CLASS_NAMES[pred_label]}",
        fontsize=16,
    )

    plt.tight_layout()

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(fig)

    print(
        f"Saved: {output_path}"
    )


def main():
    print()
    print(
        "Aegis AI - Visual Error Analysis"
    )
    print("=" * 70)

    device = get_device()

    print("Device:", device)

    dataset = XView2BuildingDataset(
        VAL_FILE
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    model = load_model(device)

    print(
        "Generating predictions..."
    )

    predictions = collect_predictions(
        dataset,
        model,
        device,
    )

    print(
        "Predictions complete."
    )

    cases = [
        (
            0,
            1,
            "no_damage_to_minor.png",
        ),
        (
            0,
            2,
            "no_damage_to_major.png",
        ),
        (
            1,
            2,
            "minor_to_major.png",
        ),
        (
            2,
            1,
            "major_to_minor.png",
        ),
        (
            3,
            1,
            "destroyed_to_minor.png",
        ),
        (
            3,
            0,
            "destroyed_to_no_damage.png",
        ),
    ]

    for (
        true_label,
        pred_label,
        filename,
    ) in cases:
        indices = select_examples(
            dataset,
            predictions,
            true_label,
            pred_label,
            limit=12,
        )

        create_grid(
            dataset,
            predictions,
            indices,
            true_label,
            pred_label,
            OUTPUT_DIR / filename,
        )

    print()
    print(
        "Visual error analysis complete."
    )

    print(
        "Output:",
        OUTPUT_DIR,
    )


if __name__ == "__main__":
    main()