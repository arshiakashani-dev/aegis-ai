from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image

from src.models.resnet_change_aware import AegisResNet18ChangeAware


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "best_resnet_change_aware.pt"
)

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


def image_to_tensor(path):
    with Image.open(path) as image:
        image = image.convert("RGB")
        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    tensor = torch.from_numpy(array)
    tensor = tensor.permute(2, 0, 1)

    return tensor


def build_input(pre_path, post_path):
    pre = image_to_tensor(pre_path)
    post = image_to_tensor(post_path)

    if pre.shape != post.shape:
        raise ValueError(
            "Pre and post images must have the same shape. "
            f"Got {pre.shape} and {post.shape}."
        )

    difference = torch.abs(pre - post)

    return torch.cat(
        [
            pre,
            post,
            difference,
        ],
        dim=0,
    )


def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=False,
    ).to(device)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    return model


def predict(pre_path, post_path):
    device = get_device()

    model = load_model(device)

    image = build_input(
        pre_path,
        post_path,
    )

    image = image.unsqueeze(0).to(device)

    with torch.inference_mode():
        logits = model(image)
        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class = int(
        torch.argmax(probabilities).item()
    )

    result = {
        "prediction": CLASS_NAMES[predicted_class],
        "confidence": float(
            probabilities[predicted_class].item()
        ),
        "probabilities": {
            name: float(probabilities[index].item())
            for index, name in enumerate(CLASS_NAMES)
        },
        "device": str(device),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Aegis AI building damage inference"
    )

    parser.add_argument(
        "--pre",
        required=True,
        help="Path to pre-disaster building crop",
    )

    parser.add_argument(
        "--post",
        required=True,
        help="Path to post-disaster building crop",
    )

    args = parser.parse_args()

    result = predict(
        Path(args.pre),
        Path(args.post),
    )

    print("\nAegis AI Prediction")
    print("=" * 40)
    print(
        f"Prediction: {result['prediction']}"
    )
    print(
        f"Confidence: {result['confidence']:.4f}"
    )

    print("\nClass probabilities:")

    for name, probability in result[
        "probabilities"
    ].items():
        print(
            f"  {name:12s} {probability:.4f}"
        )

    print(
        f"\nDevice: {result['device']}"
    )


if __name__ == "__main__":
    main()
