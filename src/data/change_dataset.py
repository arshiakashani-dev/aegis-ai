import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


PROJECT_ROOT = Path(__file__).resolve().parents[2]


DAMAGE_CLASSES = {
    "no-damage": 0,
    "minor-damage": 1,
    "major-damage": 2,
    "destroyed": 3,
}


class XView2ChangeDataset(Dataset):
    """
    XView2 building-level dataset with explicit change representation.

    Input channels:
        0:3  -> Pre-disaster RGB
        3:6  -> Post-disaster RGB
        6:9  -> Absolute RGB difference |Pre - Post|
    """

    def __init__(self, manifest_file):
        self.manifest_file = Path(manifest_file)

        self.split = (
            self.manifest_file.stem
        )

        self.cache_dir = (
            PROJECT_ROOT
            / "data"
            / "processed"
            / "crops"
            / self.split
        )

        self.rows = self._load_manifest()

    def _load_manifest(self):
        with open(
            self.manifest_file,
            "r",
            newline="",
        ) as file:
            return list(
                csv.DictReader(file)
            )

    def __len__(self):
        return len(self.rows)

    @staticmethod
    def _image_to_tensor(image):
        array = np.asarray(
            image,
            dtype=np.float32,
        ) / 255.0

        tensor = torch.from_numpy(
            array
        )

        tensor = tensor.permute(
            2,
            0,
            1,
        )

        return tensor

    def __getitem__(self, index):
        row = self.rows[index]

        uid = row["uid"]
        scene_id = row["scene_id"]

        sample_dir = (
            self.cache_dir
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

        if not pre_path.exists():
            raise FileNotFoundError(
                pre_path
            )

        if not post_path.exists():
            raise FileNotFoundError(
                post_path
            )

        with Image.open(
            pre_path
        ) as image:
            pre_image = image.convert(
                "RGB"
            )

        with Image.open(
            post_path
        ) as image:
            post_image = image.convert(
                "RGB"
            )

        pre_tensor = (
            self._image_to_tensor(
                pre_image
            )
        )

        post_tensor = (
            self._image_to_tensor(
                post_image
            )
        )

        difference = torch.abs(
            pre_tensor
            - post_tensor
        )

        image = torch.cat(
            [
                pre_tensor,
                post_tensor,
                difference,
            ],
            dim=0,
        )

        label = DAMAGE_CLASSES[
            row["damage_label"]
        ]

        return (
            image,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )