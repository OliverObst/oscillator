"""Explicit root-path decomposition and provisional contact phase supervision."""

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation

from .data import DATASET_ID, DATASET_REVISION, MotionClip, load_clips

DEFAULT_SPLITS = {
    "train": ["run2_subject1_05", "walk3_subject2_01", "walk3_subject2_02"],
    "dev": ["run2_subject1_06", "walk3_subject2_03"],
    "test": ["run2_subject1_07", "walk3_subject2_04"],
}


@dataclass(frozen=True)
class PreprocessConfig:
    path_sigma_s: float = 0.6
    contact_height_m: float = 0.045
    contact_speed_m_s: float = 0.6
    min_cycle_s: float = 0.35
    max_cycle_s: float = 2.0
    unequal_halves: bool = False

    def __post_init__(self):
        if self.path_sigma_s <= 0 or self.contact_height_m <= 0 or self.contact_speed_m_s <= 0:
            raise ValueError("Path smoothing and contact thresholds must be positive")
        if not 0 < self.min_cycle_s < self.max_cycle_s:
            raise ValueError("Invalid cycle duration bounds")


def yaw_rotation(yaw: np.ndarray) -> Rotation:
    yaw = np.asarray(yaw)
    return Rotation.from_rotvec(np.stack([np.zeros_like(yaw), np.zeros_like(yaw), yaw], -1))


def fit_root_path(clip: MotionClip, sigma_s: float) -> tuple[np.ndarray, np.ndarray]:
    """Offline fixed-bandwidth fit; only planar position and unwrapped heading are fitted."""
    if sigma_s <= 0:
        raise ValueError("Path sigma must be positive")
    root_rotation = Rotation.from_quat(clip.root_rot_xyzw)
    forward = root_rotation.as_matrix()[:, :, 0]
    heading = np.unwrap(np.arctan2(forward[:, 1], forward[:, 0]))
    path_yaw = gaussian_filter1d(heading, sigma_s * clip.fps, mode="nearest")
    path_pos = np.zeros_like(clip.root_pos)
    path_pos[:, :2] = gaussian_filter1d(
        clip.root_pos[:, :2],
        sigma_s * clip.fps,
        axis=0,
        mode="nearest",
    )
    return path_pos, path_yaw


def decompose_root(root_pos, root_rot_xyzw, path_pos, path_yaw) -> np.ndarray:
    path_rotation = yaw_rotation(path_yaw)
    translation = path_rotation.inv().apply(root_pos - path_pos)
    rotation_vector = (path_rotation.inv() * Rotation.from_quat(root_rot_xyzw)).as_rotvec()
    return np.concatenate([translation, rotation_vector], axis=1)


def reconstruct_root(residual, path_pos, path_yaw) -> tuple[np.ndarray, np.ndarray]:
    path_rotation = yaw_rotation(path_yaw)
    position = path_pos + path_rotation.apply(residual[:, :3])
    rotation = path_rotation * Rotation.from_rotvec(residual[:, 3:6])
    return position, rotation.as_quat()


def world_feet(clip: MotionClip) -> np.ndarray:
    indices = []
    for side in ("left", "right"):
        matches = [
            i
            for i, name in enumerate(clip.link_body_list)
            if side in name.lower() and "foot" in name.lower()
        ]
        if len(matches) != 1:
            raise ValueError(f"{clip.name}: expected one {side} foot body, found {matches}")
        indices.append(matches[0])
    local = clip.local_body_pos[:, indices]
    rotation = Rotation.from_quat(np.repeat(clip.root_rot_xyzw, 2, axis=0))
    return rotation.apply(local.reshape(-1, 3)).reshape(-1, 2, 3) + clip.root_pos[:, None]


def contact_labels(feet, fps: float, config: PreprocessConfig) -> np.ndarray:
    smooth = gaussian_filter1d(feet, 0.025 * fps, axis=0, mode="nearest")
    speed = np.linalg.norm(np.gradient(smooth, 1 / fps, axis=0), axis=2)
    floor = np.quantile(smooth[:, :, 2], 0.1, axis=0)
    return (smooth[:, :, 2] <= floor + config.contact_height_m) & (
        speed <= config.contact_speed_m_s
    )


def onsets(contact: np.ndarray) -> np.ndarray:
    # Do not invent a strike at frame zero when a clip starts during stance.
    return np.flatnonzero(np.diff(contact.astype(int), prepend=int(contact[0])) == 1)


def phase_labels(feet, contacts, fps: float, config: PreprocessConfig):
    strikes = onsets(contacts[:, 0])
    source = "left_contact_onsets"
    durations = np.diff(strikes) / fps
    good = (durations >= config.min_cycle_s) & (durations <= config.max_cycle_s)
    if good.sum() < 3:
        # Provisional fallback; low foot-height extrema are not measured contacts.
        height = gaussian_filter1d(feet[:, 0, 2], 0.04 * fps)
        strikes, _ = find_peaks(
            -height, distance=max(1, int(config.min_cycle_s * fps)), prominence=0.015
        )
        source = "left_foot_height_minima_fallback"
    phase = np.full(len(feet), np.nan)
    phase_rate = np.full(len(feet), np.nan)
    period = np.full(len(feet), np.nan)
    cycles = []
    right = onsets(contacts[:, 1])
    for start, end in zip(strikes[:-1], strikes[1:], strict=True):
        duration = (end - start) / fps
        if not config.min_cycle_s <= duration <= config.max_cycle_s:
            continue
        halves = right[(right > start) & (right < end)]
        if config.unequal_halves:
            if len(halves) != 1:
                continue
            middle = int(halves[0])
            if min(middle - start, end - middle) / fps < 0.1:
                continue
            phase[start:middle] = 0.5 * np.arange(middle - start) / (middle - start)
            phase[middle:end] = 0.5 + 0.5 * np.arange(end - middle) / (end - middle)
            phase_rate[start:middle] = 0.5 * fps / (middle - start)
            phase_rate[middle:end] = 0.5 * fps / (end - middle)
        else:
            phase[start:end] = np.arange(end - start) / (end - start)
            phase_rate[start:end] = 1 / duration
        period[start:end] = duration
        cycles.append(
            {
                "start": int(start),
                "end": int(end),
                "period_s": duration,
                "right_strikes": halves.tolist(),
            }
        )
    if len(cycles) < 3:
        raise ValueError("Fewer than three plausible complete cycles; inspect contact thresholds")
    return phase, phase_rate, period, cycles, source


def prepare_clip(clip: MotionClip, config: PreprocessConfig):
    path_pos, path_yaw = fit_root_path(clip, config.path_sigma_s)
    residual = decompose_root(clip.root_pos, clip.root_rot_xyzw, path_pos, path_yaw)
    reconstructed, quat = reconstruct_root(residual, path_pos, path_yaw)
    error_pos = float(np.max(np.abs(reconstructed - clip.root_pos)))
    error_rot = float(
        np.max(
            (Rotation.from_quat(quat).inv() * Rotation.from_quat(clip.root_rot_xyzw)).magnitude()
        )
    )
    feet = world_feet(clip)
    contacts = contact_labels(feet, clip.fps, config)
    try:
        phase, rate, period, cycles, source = phase_labels(feet, contacts, clip.fps, config)
    except ValueError as error:
        raise ValueError(f"{clip.name}: {error}") from error
    velocity = yaw_rotation(path_yaw).inv().apply(np.gradient(path_pos, 1 / clip.fps, axis=0))
    yaw_rate = np.gradient(path_yaw, 1 / clip.fps)
    context = np.full((len(phase), 4), np.nan)
    cycle_id = np.full(len(phase), -1, dtype=int)
    for i, cycle in enumerate(cycles):
        sl = slice(cycle["start"], cycle["end"])
        context[sl] = [*velocity[sl, :2].mean(0), yaw_rate[sl].mean(), cycle["period_s"]]
        cycle_id[sl] = i
    valid = np.isfinite(phase)
    arrays = {
        "time": clip.time,
        "fps": np.array(clip.fps),
        "phase": phase,
        "phase_rate": rate,
        "period": period,
        "context": context,
        "valid": valid,
        "cycle_id": cycle_id,
        "target": np.concatenate([clip.dof_pos, residual], axis=1),
        "path_pos": path_pos,
        "path_yaw": path_yaw,
        "root_pos": clip.root_pos,
        "root_rot_xyzw": clip.root_rot_xyzw,
        "contacts": contacts,
        "feet_world": feet,
    }
    report = {
        "name": clip.name,
        "frames": len(valid),
        "valid_frames": int(valid.sum()),
        "fps": clip.fps,
        "phase_source": source,
        "cycles": cycles,
        "contact_fraction": contacts.mean(0).tolist(),
        "root_roundtrip_max_m": error_pos,
        "root_roundtrip_max_rad": error_rot,
    }
    return arrays, report


def validate_splits(splits, names):
    if set(splits) != {"train", "dev", "test"} or any(not v for v in splits.values()):
        raise ValueError("Specify non-empty train, dev and test clip lists")
    assigned = [name for group in splits.values() for name in group]
    if len(assigned) != len(set(assigned)) or set(assigned) != set(names):
        raise ValueError("Splits must assign every clip exactly once")


def prepare_dataset(
    parquet: Path, output: Path, config=None, splits=None, revision=DATASET_REVISION
):
    config = PreprocessConfig() if config is None else config
    clips = load_clips(parquet)
    splits = DEFAULT_SPLITS if splits is None else splits
    validate_splits(splits, [c.name for c in clips])
    # Prepare all clips before writing, so a schema/phase failure leaves no partial dataset.
    prepared = [prepare_clip(clip, config) for clip in clips]
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for clip, (arrays, report) in zip(clips, prepared, strict=True):
        np.savez_compressed(output / f"{clip.name}.npz", **arrays)
        reports.append(report)
    import hashlib

    metadata = {
        "schema_version": 1,
        "dataset_id": DATASET_ID,
        "dataset_revision": revision,
        "source_sha256": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "config": asdict(config),
        "splits": splits,
        "joint_names": clips[0].joint_names,
        "coordinate_names": clips[0].joint_names
        + [
            "root_dx",
            "root_dy",
            "root_height",
            "root_rx",
            "root_ry",
            "root_rz",
        ],
        "clips": reports,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return metadata


def read_prepared(directory: Path):
    metadata = json.loads((directory / "metadata.json").read_text())
    clips = {}
    for report in metadata["clips"]:
        with np.load(directory / f"{report['name']}.npz", allow_pickle=False) as archive:
            clips[report["name"]] = {k: archive[k] for k in archive.files}
    validate_splits(metadata["splits"], clips)
    return metadata, clips
