import json

import numpy as np
import pytest
import torch

from oscillator.cli import compatible_metadata
from oscillator.preprocess import yaw_rotation
from oscillator.train import TrainConfig, load_checkpoint, train


def write_tiny_dataset(directory):
    directory.mkdir()
    n = 96
    time = np.arange(n) / 30
    phase = (time / 0.8) % 1
    context = np.tile([0.8, 0, 0.05, 0.8], (n, 1))
    context[24:48, 0] = 1.2
    context[48:, 0] = 1.4
    target = np.sin(2 * np.pi * phase[:, None]) * np.linspace(0.01, 0.2, 28)[None]
    target[:, 24] += 0.55
    path_pos = np.column_stack([time * 0.8, np.zeros(n), np.zeros(n)])
    path_yaw = time * 0.05
    for name, offset in (("train", 0.0), ("dev", 0.01), ("test", 0.1)):
        y = target + offset
        root = path_pos + yaw_rotation(path_yaw).apply(y[:, 22:25])
        np.savez_compressed(
            directory / f"{name}.npz",
            time=time,
            phase=phase,
            context=context,
            target=y,
            valid=np.ones(n, dtype=bool),
            fps=30,
            path_pos=path_pos,
            path_yaw=path_yaw,
            root_pos=root,
            root_rot_xyzw=yaw_rotation(path_yaw).as_quat(),
        )
    metadata = {
        "schema_version": 1,
        "source_sha256": "synthetic",
        "config": {},
        "joint_names": [f"joint_{i}" for i in range(22)],
        "splits": {name: [name] for name in ("train", "dev", "test")},
        "clips": [{"name": name} for name in ("train", "dev", "test")],
    }
    (directory / "metadata.json").write_text(json.dumps(metadata))
    return metadata, target


def test_training_checkpoint_and_holdout_isolation(tmp_path):
    data = tmp_path / "data"
    metadata, train_target = write_tiny_dataset(data)
    config = TrainConfig(epochs=4, batch_size=48, cadence=True, tau_candidates=(0.1, 0.2))
    first_report = train(data, tmp_path / "first", config)
    model, cadence, checkpoint = load_checkpoint(tmp_path / "first/model.pt")
    assert cadence is not None
    assert first_report["decoder_parameters"] == 2328
    assert first_report["cadence_parameters"] == 81
    compatible_metadata(metadata, checkpoint)
    np.testing.assert_allclose(model.target_mean.numpy(), train_target.mean(0), atol=1e-7)
    with np.load(data / "test.npz") as source:
        payload = {k: source[k] for k in source.files}
    payload["target"] += 100
    np.savez_compressed(data / "test.npz", **payload)
    second_report = train(data, tmp_path / "second", config)
    second, _, second_checkpoint = load_checkpoint(tmp_path / "second/model.pt")
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[key], atol=0, rtol=0)
    assert checkpoint["tau_z_s"] == second_checkpoint["tau_z_s"]
    assert first_report["best_epoch"] == second_report["best_epoch"]
    assert first_report["metrics"]["test"] != second_report["metrics"]["test"]
    changed = dict(metadata, source_sha256="different")
    with pytest.raises(ValueError, match="source_sha256"):
        compatible_metadata(changed, checkpoint)
