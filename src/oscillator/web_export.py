"""Export the fitted decoder and source clips for a static, offline browser viewer."""

import json
from pathlib import Path

import numpy as np
import torch

from .cli import compatible_metadata
from .preprocess import read_prepared
from .runtime import OscillatorRuntime
from .train import load_checkpoint

FRAME_FIELDS = [
    ("dof_pos", 22),
    ("root_pos", 3),
    ("root_rot_xyzw", 4),
    ("path_pos", 3),
    ("path_yaw", 1),
    ("phase", 1),
    ("context", 4),
    ("valid", 1),
    ("contacts", 2),
]


def export_web(prepared: Path, checkpoint_path: Path, output: Path):
    model, cadence, checkpoint = load_checkpoint(checkpoint_path)
    metadata, clips = read_prepared(prepared)
    compatible_metadata(metadata, checkpoint)
    output.mkdir(parents=True, exist_ok=True)
    state = model.state_dict()
    serialised = {
        "schema_version": 1,
        "joint_names": checkpoint["joint_names"],
        "waveforms": state["waveforms"].tolist(),
        "tau_z_s": checkpoint["tau_z_s"],
        "layers": [
            {
                "weight": state[f"context_net.{i}.weight"].tolist(),
                "bias": state[f"context_net.{i}.bias"].tolist(),
                "activation": "tanh" if i < 4 else "linear",
            }
            for i in (0, 2, 4)
        ],
        **{
            key: state[key].tolist()
            for key in ("context_mean", "context_scale", "target_mean", "target_scale")
        },
        "parameters": model.parameter_count,
        "dataset_id": metadata["dataset_id"],
        "dataset_revision": metadata["dataset_revision"],
        "source_sha256": metadata["source_sha256"],
        "cadence": None,
    }
    if cadence is not None:
        serialised["cadence"] = {
            "min_hz": cadence.min_hz,
            "max_hz": cadence.max_hz,
            "mean": cadence.command_mean.tolist(),
            "scale": cadence.command_scale.tolist(),
            "layers": [
                {
                    "weight": cadence.net[i].weight.tolist(),
                    "bias": cadence.net[i].bias.tolist(),
                    "activation": "tanh" if i == 0 else "linear",
                }
                for i in (0, 2)
            ],
        }
    (output / "model.json").write_text(json.dumps(serialised, separators=(",", ":")) + "\n")
    report_path = checkpoint_path.parent / "report.json"
    metrics = json.loads(report_path.read_text())["metrics"] if report_path.exists() else {}
    manifest = {
        "schema_version": 1,
        "fields": FRAME_FIELDS,
        "stride": 41,
        "dtype": "little-endian float32",
        "clips": [],
    }
    for summary in metadata["clips"]:
        name, clip = summary["name"], clips[summary["name"]]
        n = len(clip["time"])
        fields = dict(clip, dof_pos=clip["target"][:, :22])
        packed = np.concatenate(
            [np.asarray(fields[key]).reshape(n, width) for key, width in FRAME_FIELDS], axis=1
        ).astype("<f4")
        packed.tofile(output / f"{name}.bin")
        split = next(s for s, names in metadata["splits"].items() if name in names)
        manifest["clips"].append(
            {
                "name": name,
                "file": f"{name}.bin",
                "frames": n,
                "fps": float(clip["fps"]),
                "duration_s": float(clip["time"][-1]),
                "split": split,
                "valid_frames": int(clip["valid"].sum()),
                "phase_source": summary["phase_source"],
                "metrics": metrics.get(split, {}).get(name, {}),
            }
        )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    # Cross-language fixtures verify browser inference against the actual saved PyTorch model.
    cases = []
    with torch.no_grad():
        for phase, context in (
            (0.0, [0.8, 0.0, 0.0, 0.9]),
            (0.37, [1.2, 0.2, -0.3, 0.75]),
            (0.99, [-0.3, -0.1, 0.4, 1.2]),
        ):
            x, phi = torch.tensor(context), torch.tensor(phase)
            z = model.mixture(x)
            z_rate = torch.tensor([0.2, -0.1, 0.3, 0.1])
            cases.append(
                {
                    "phase": phase,
                    "context": context,
                    "z": z.tolist(),
                    "y": model(phi, x).tolist(),
                    "z_rate": z_rate.tolist(),
                    "phase_rate": 1.3,
                    "dy": model.reference_derivative(phi, z, torch.tensor(1.3), z_rate).tolist(),
                }
            )
    runtime = OscillatorRuntime(model, checkpoint["tau_z_s"], cadence)
    commands = [[0.8, 0, 0, 0.9], [1.2, 0.2, 0.3, 0.7]]
    runtime_state = runtime.initial_state(commands[0])
    frames = []
    for i in range(60):
        ref = runtime.advance(runtime_state, commands[0 if i < 30 else 1], 1 / 60)
        if i in (0, 29, 30, 59):
            frames.append(
                {
                    "step": i + 1,
                    "phase": runtime_state.phase,
                    "z": runtime_state.z.tolist(),
                    "y": ref.coordinates.tolist(),
                    "root_pos": ref.root_pos.tolist(),
                    "root_rot_xyzw": ref.root_rot_xyzw.tolist(),
                }
            )
    kinematics = []
    for name in (metadata["clips"][0]["name"], metadata["clips"][-1]["name"]):
        clip = clips[name]
        for index in (0, 50, 150):
            kinematics.append(
                {
                    "joints": clip["target"][index, :22].tolist(),
                    "position": clip["root_pos"][index].tolist(),
                    "quaternion": clip["root_rot_xyzw"][index].tolist(),
                    "feet": clip["feet_world"][index].tolist(),
                }
            )
    (output / "validation.json").write_text(
        json.dumps({"cases": cases, "runtime": frames, "kinematics": kinematics}) + "\n"
    )
    return manifest
