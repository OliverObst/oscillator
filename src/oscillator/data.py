"""Download the pinned Parquet source and validate its motion schema."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

DATASET_ID = "whirlwind-ams/lafan_locomotion_k1"
DATASET_REVISION = "8cb1332286a7d28d3f9ffc4c27bb2de3b861cbae"


@dataclass
class MotionClip:
    name: str
    fps: float
    joint_names: list[str]
    root_pos: np.ndarray
    root_rot_xyzw: np.ndarray
    dof_pos: np.ndarray
    local_body_pos: np.ndarray
    link_body_list: list[str]

    def __post_init__(self):
        n = len(self.root_pos)
        if not np.isfinite(self.fps) or self.fps <= 0 or n < 4:
            raise ValueError(f"{self.name}: invalid fps or clip length")
        expected = {
            "root_pos": (n, 3),
            "root_rot_xyzw": (n, 4),
            "dof_pos": (n, 22),
            "local_body_pos": (n, len(self.link_body_list), 3),
        }
        for key, shape in expected.items():
            value = np.asarray(getattr(self, key), dtype=np.float64)
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError(f"{self.name}: {key} must be finite with shape {shape}")
            setattr(self, key, value)
        if len(self.joint_names) != 22 or len(set(self.joint_names)) != 22:
            raise ValueError(f"{self.name}: expected 22 unique joint names")
        norms = np.linalg.norm(self.root_rot_xyzw, axis=1)
        if np.any(norms < 1e-6) or np.any(np.abs(norms - 1) > 0.05):
            raise ValueError(f"{self.name}: invalid root quaternions")
        self.root_rot_xyzw /= norms[:, None]

    @property
    def time(self):
        return np.arange(len(self.root_pos)) / self.fps


def load_clips(parquet: Path) -> list[MotionClip]:
    clips = [
        MotionClip(**{k: row[k] for k in MotionClip.__dataclass_fields__})
        for row in pq.read_table(parquet).to_pylist()
    ]
    if not clips or len({c.name for c in clips}) != len(clips):
        raise ValueError("Expected non-empty, uniquely named clips")
    if any(c.joint_names != clips[0].joint_names for c in clips):
        raise ValueError("All clips must share the same ordered joint names")
    return clips


def download_dataset(revision: str = DATASET_REVISION) -> Path:
    return Path(
        hf_hub_download(
            DATASET_ID,
            "data/motions.parquet",
            repo_type="dataset",
            revision=revision,
        )
    )
