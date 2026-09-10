from pathlib import Path
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.change_dataset import XView2ChangeDataset
from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

VAL_MANIFEST = PROJECT_ROOT / "data" / "processed" / "val.csv"
CHECKPOINT = PROJECT_ROOT / "checkpoints" / "best_resnet_change_aware.pt"

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "error_analysis_resnet"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


CLASS_NAMES = [
    "no-damage",
    "minor-damage",
    "major-damage",
    "destroyed",
]

CLASS_TO_ID = {
    name: index
    for index, name in enumerate(CLASS_NAMES)
}


ERROR_TYPES = [
    ("no-damage", "minor-damage"),
    ("no-damage", "major-damage"),
    ("minor-damage", "major-damage"),
    ("major-damage", "minor-damage"),
    ("destroyed", "minor-damage"),
    ("destroyed", "no-damage"),
]

MAX_EXAMPLES_PER_ERROR = 12
BATCH_SIZE = 16


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def tensor_to_image(tensor):
    image = tensor.detach().cpu().permute(1, 2, 0).numpy()
    return np.clip(image, 0.0, 1.0)


def collect_errors(model, dataset, device):
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    target_pairs = {
        (
            CLASS_TO_ID[true_name],
            CLASS_TO_ID[pred_name],
        )
        for true_name, pred_name in ERROR_TYPES
    }

    found = defaultdict(list)

    offset = 0

    model.eval()

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)

            outputs = model(images)
            predictions = outputs.argmax(dim=1).cpu()

            for batch_index in range(labels.size(0)):
                true_id = labels[batch_index].item()
                pred_id = predictions[batch_index].item()

                pair = (true_id, pred_id)

                if pair in target_pairs:
                    if len(found[pair]) < MAX_EXAMPLES_PER_ERROR:
                        found[pair].append(offset + batch_index)

            offset += labels.size(0)

            complete = all(
                len(found[pair]) >= MAX_EXAMPLES_PER_ERROR
                for pair in target_pairs
            )

            if complete:
                break

    return found


def save_error_grid(
    dataset,
    indices,
    true_name,
    predicted_name,
):
    if not indices:
        print(
            f"No examples found for "
            f"{true_name} -> {predicted_name}"
        )
        return

    rows = len(indices)
    columns = 3

    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(10, 3.2 * rows),
    )

    if rows == 1:
        axes = np.expand_dims(axes, axis=0)

    for row_index, dataset_index in enumerate(indices):
        image, _ = dataset[dataset_index]

        pre = image[0:3]
        post = image[3:6]
        difference = image[6:9]

        metadata = dataset.rows[dataset_index]

        scene_id = metadata["scene_id"]
        uid = metadata["uid"]

        axes[row_index, 0].imshow(
            tensor_to_image(pre)
        )

        axes[row_index, 1].imshow(
            tensor_to_image(post)
        )

        axes[row_index, 2].imshow(
            tensor_to_image(difference)
        )

        axes[row_index, 0].set_title(
            f"PRE\n{scene_id}\n{uid[:12]}"
        )

        axes[row_index, 1].set_title(
            f"POST\nGT: {true_name}\nPred: {predicted_name}"
        )

        axes[row_index, 2].set_title(
            "|PRE - POST|"
        )

        for column in range(columns):
            axes[row_index, column].axis("off")

    figure.suptitle(
        f"ResNet18 Errors: "
        f"{true_name} → {predicted_name}",
        fontsize=16,
    )

    figure.tight_layout(
        rect=[0, 0, 1, 0.985]
    )

    filename = (
        f"{true_name.replace('-', '_')}"
        f"_to_"
        f"{predicted_name.replace('-', '_')}.png"
    )

    output_path = OUTPUT_DIR / filename

    figure.savefig(
        output_path,
        dpi=150,
        bbox_inches="tight",
    )

    plt.close(figure)

    print(
        f"Saved {len(indices):2d} examples -> "
        f"{output_path}"
    )


def main():
    print(
        "Aegis AI - ResNet18 Visual Error Analysis"
    )
    print("=" * 60)

    device = get_device()
    print("Device:", device)

    dataset = XView2ChangeDataset(
        VAL_MANIFEST
    )

    print(
        "Validation samples:",
        len(dataset),
    )

    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    print(
        "Checkpoint epoch:",
        checkpoint["epoch"],
    )

    print(
        "Checkpoint Macro F1:",
        f"{checkpoint['val_macro_f1']:.4f}",
    )

    print("\nSearching for error examples...")

    found = collect_errors(
        model,
        dataset,
        device,
    )

    print("\nGenerating visual grids...")

    for true_name, predicted_name in ERROR_TYPES:
        pair = (
            CLASS_TO_ID[true_name],
            CLASS_TO_ID[predicted_name],
        )

        save_error_grid(
            dataset=dataset,
            indices=found[pair],
            true_name=true_name,
            predicted_name=predicted_name,
        )

    print("\n" + "=" * 60)
    print("Visual error analysis complete.")
    print("Output directory:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
