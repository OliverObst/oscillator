import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from oscillator.data import MotionClip
from oscillator.preprocess import (
    PreprocessConfig,
    decompose_root,
    fit_root_path,
    phase_labels,
    reconstruct_root,
    validate_splits,
    world_feet,
    yaw_rotation,
)


def test_root_roundtrip_through_multiple_yaw_wraps():
    n = 400
    yaw = np.linspace(-4, 10, n)
    path_pos = np.column_stack([np.linspace(0, 5, n), np.sin(yaw), np.zeros(n)])
    residual = np.column_stack(
        [
            0.05 * np.sin(4 * yaw),
            0.03 * np.cos(yaw),
            np.full(n, 0.55),
            0.1 * np.sin(yaw),
            0.1 * np.cos(yaw),
            0.03 * np.sin(3 * yaw),
        ]
    )
    position, quat = reconstruct_root(residual, path_pos, yaw)
    # Flip quaternion signs: a rotation must not depend on quaternion sign.
    quat[::2] *= -1
    recovered = decompose_root(position, quat, path_pos, yaw)
    np.testing.assert_allclose(recovered, residual, atol=1e-12)
    position2, quat2 = reconstruct_root(recovered, path_pos, yaw)
    np.testing.assert_allclose(position2, position, atol=1e-12)
    np.testing.assert_allclose(
        (Rotation.from_quat(quat).inv() * Rotation.from_quat(quat2)).magnitude(), 0, atol=1e-12
    )


def synthetic_clip():
    n = 300
    time = np.arange(n) / 30
    yaw = 2.8 + 0.2 * time + 0.1 * np.sin(2 * np.pi * time)
    root = np.column_stack([time, np.zeros(n), 0.5 + 0.02 * np.sin(2 * np.pi * time)])
    feet = np.tile([[0.2, 0.1, -0.5], [-0.2, -0.1, -0.5]], (n, 1, 1))
    return MotionClip(
        "synthetic",
        30,
        [f"joint_{i}" for i in range(22)],
        root,
        yaw_rotation(yaw).as_quat(),
        np.zeros((n, 22)),
        feet,
        ["left_foot_link", "right_foot_link"],
    )


def test_path_is_planar_slow_and_preserves_height_in_residual():
    clip = synthetic_clip()
    path, yaw = fit_root_path(clip, 0.6)
    assert np.all(path[:, 2] == 0)
    assert np.max(np.abs(np.diff(yaw))) < 0.02
    residual = decompose_root(clip.root_pos, clip.root_rot_xyzw, path, yaw)
    np.testing.assert_allclose(residual[:, 2], clip.root_pos[:, 2])
    # The path retains progression rather than independently aligning each frame.
    assert path[-1, 0] - path[0, 0] > 9
    assert np.std(residual[:, 5]) > 0.04


def test_world_feet_apply_root_orientation_and_translation():
    clip = synthetic_clip()
    actual = world_feet(clip)
    expected = Rotation.from_quat(clip.root_rot_xyzw).apply(clip.local_body_pos[:, 0])
    np.testing.assert_allclose(actual[:, 0], expected + clip.root_pos)


@pytest.mark.parametrize("unequal", [False, True])
def test_phase_complete_cycles_and_explicit_unequal_halves(unequal):
    n, fps = 220, 30
    contacts = np.zeros((n, 2), dtype=bool)
    for start in range(10, 211, 40):
        contacts[start : start + 4, 0] = True
        contacts[start + 15 : start + 19, 1] = True
    phi, rate, period, cycles, source = phase_labels(
        np.zeros((n, 2, 3)),
        contacts,
        fps,
        PreprocessConfig(unequal_halves=unequal),
    )
    assert source == "left_contact_onsets"
    assert len(cycles) == 5
    assert np.isnan(phi[:10]).all() and np.isnan(phi[210:]).all()
    assert phi[10] == 0 and period[10] == 40 / fps
    if unequal:
        assert phi[25] == 0.5
        assert rate[10] == 0.5 * fps / 15
        assert rate[25] == 0.5 * fps / 25
    else:
        assert phi[25] == 15 / 40
        assert rate[10] == fps / 40


def test_splits_reject_leakage_and_unassigned_clips():
    validate_splits({"train": ["a"], "dev": ["b"], "test": ["c"]}, ["a", "b", "c"])
    with pytest.raises(ValueError, match="exactly once"):
        validate_splits({"train": ["a", "b"], "dev": ["b"], "test": ["c"]}, ["a", "b", "c"])
    with pytest.raises(ValueError, match="exactly once"):
        validate_splits({"train": ["a"], "dev": ["b"], "test": ["c"]}, ["a", "b", "c", "d"])
