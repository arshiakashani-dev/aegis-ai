from pathlib import Path
import argparse

import numpy as np
import torch
from PIL import Image

from src.models.resnet_change_aware import (
    AegisResNet18ChangeAware,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / "checkpoints"
    / "context32_aug_resnet_change_aware.pt"
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
    path = Path(path)

    if not path.exists():
        raise FileNotFoundError(
            f"Image not found: {path}"
        )

    with Image.open(path) as image:
        image = image.convert("RGB")

        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

    tensor = torch.from_numpy(array)

    tensor = tensor.permute(
        2,
        0,
        1,
    )

    return tensor


def build_input(pre_path, post_path):
    pre = image_to_tensor(pre_path)
    post = image_to_tensor(post_path)

    if pre.shape != post.shape:
        raise ValueError(
            "Pre and post images must have the same shape. "
            f"Got {pre.shape} and {post.shape}."
        )

    difference = torch.abs(
        pre - post
    )

    image = torch.cat(
        [
            pre,
            post,
            difference,
        ],
        dim=0,
    )

    if image.shape[0] != 9:
        raise ValueError(
            "Aegis AI Context32 model expects "
            f"9 input channels, got {image.shape[0]}."
        )

    return image


def load_model(device):
    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: "
            f"{CHECKPOINT_PATH}"
        )

    model = AegisResNet18ChangeAware(
        num_classes=4,
        pretrained=False,
    )

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=device,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint[
            "model_state_dict"
        ]
    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.to(device)
    model.eval()

    return model


def predict(pre_path, post_path):
    device = get_device()

    model = load_model(device)

    image = build_input(
        pre_path,
        post_path,
    )

    image = (
        image
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        logits = model(image)

        probabilities = torch.softmax(
            logits,
            dim=1,
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    prediction = CLASS_NAMES[
        predicted_class
    ]

    confidence = float(
        probabilities[
            predicted_class
        ].item()
    )

    result = {
        "prediction": prediction,
        "confidence": confidence,
        "probabilities": {
            name: float(
                probabilities[index].item()
            )
            for index, name
            in enumerate(CLASS_NAMES)
        },
        "device": str(device),
        "checkpoint": str(
            CHECKPOINT_PATH
        ),
    }

    return result


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Aegis AI Context32 "
            "building damage inference"
        )
    )

    parser.add_argument(
        "--pre",
        required=True,
        help=(
            "Path to pre-disaster "
            "building crop"
        ),
    )

    parser.add_argument(
        "--post",
        required=True,
        help=(
            "Path to post-disaster "
            "building crop"
        ),
    )

    args = parser.parse_args()

    result = predict(
        Path(args.pre),
        Path(args.post),
    )

    print()
    print(
        "Aegis AI - Context32 Inference"
    )
    print("=" * 50)

    print(
        f"Prediction: "
        f"{result['prediction']}"
    )

    print(
        f"Confidence: "
        f"{result['confidence']:.4f}"
    )

    print()
    print("Class probabilities")
    print("-" * 50)

    for name, probability in (
        result["probabilities"].items()
    ):
        print(
            f"{name:15s} "
            f"{probability:.4f}"
        )

    print()
    print(
        f"Device: "
        f"{result['device']}"
    )

    print(
        f"Checkpoint: "
        f"{result['checkpoint']}"
    )


if __name__ == "__main__":
    main()
