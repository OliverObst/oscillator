"""Small, reproducible command-line workflow."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .data import DATASET_REVISION, download_dataset, load_clips
from .preprocess import (
    PreprocessConfig,
    prepare_dataset,
    read_prepared,
    reconstruct_root,
)
from .runtime import OscillatorRuntime, StepTimings
from .train import TrainConfig, evaluate, load_checkpoint, train


def compatible_metadata(metadata, checkpoint):
    for key in ("source_sha256", "config", "joint_names", "splits"):
        if metadata[key] != checkpoint["preprocessing"][key]:
            raise ValueError(f"Prepared data and checkpoint disagree on {key}")


def reconstruct(args):
    model, _, checkpoint = load_checkpoint(args.checkpoint)
    metadata, clips = read_prepared(args.data)
    compatible_metadata(metadata, checkpoint)
    if args.clip not in clips:
        raise ValueError(f"Unknown clip {args.clip}; choose from {', '.join(clips)}")
    clip = clips[args.clip]
    mask = clip["valid"]
    with torch.no_grad():
        prediction = model(
            torch.tensor(clip["phase"][mask], dtype=torch.float32),
            torch.tensor(clip["context"][mask], dtype=torch.float32),
        ).numpy()
    root_pos, quat = reconstruct_root(
        prediction[:, 22:], clip["path_pos"][mask], clip["path_yaw"][mask]
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=clip["time"][mask],
        frame_indices=np.flatnonzero(mask),
        phase=clip["phase"][mask],
        context=clip["context"][mask],
        coordinates=prediction,
        dof_pos=prediction[:, :22],
        root_pos=root_pos,
        root_rot_xyzw=quat,
        target=clip["target"][mask],
        path_pos=clip["path_pos"][mask],
        path_yaw=clip["path_yaw"][mask],
        contacts=clip["contacts"][mask],
        joint_names=metadata["joint_names"],
    )
    if args.plot:
        plot_reconstruction(args.plot, args.clip, clip, prediction, root_pos, metadata)
    print(f"Saved reconstruction to {args.output}")


def plot_reconstruction(path, name, clip, prediction, root_pos, metadata):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask = clip["valid"]
    time = clip["time"][mask]
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), layout="constrained")
    for ax, key in zip(
        axes.flat[:4],
        ["Left_Hip_Pitch", "Right_Hip_Pitch", "Left_Knee_Pitch", "Left_Shoulder_Pitch"],
        strict=True,
    ):
        index = metadata["joint_names"].index(key)
        # Break plotted lines at excluded intervals instead of joining unrelated cycles.
        target = clip["target"][:, index]
        fitted = np.full(len(mask), np.nan)
        fitted[mask] = prediction[:, index]
        ax.plot(clip["time"], target, lw=1, label="data")
        ax.plot(clip["time"], fitted, lw=1, label="decoder")
        ax.set(title=key, xlabel="time (s)", ylabel="angle (rad)")
        ax.legend()
    ax = axes[2, 0]
    ax.plot(time, clip["root_pos"][mask, 2], label="data")
    ax.plot(time, root_pos[:, 2], label="decoder")
    ax.set(title="Root height", xlabel="time (s)", ylabel="height (m)")
    ax.legend()
    ax = axes[2, 1]
    ax.plot(clip["root_pos"][:, 0], clip["root_pos"][:, 1], label="data")
    ax.plot(clip["path_pos"][:, 0], clip["path_pos"][:, 1], label="fitted planar path")
    ax.plot(root_pos[:, 0], root_pos[:, 1], label="reconstruction")
    ax.set(title="World root trajectory", xlabel="x (m)", ylabel="y (m)", aspect="equal")
    ax.legend()
    fig.suptitle(f"{name}: observed-phase reconstruction (fixed fitted root path)")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)


def rollout(args):
    if args.duration <= 0 or args.fps <= 0 or not np.isfinite([args.duration, args.fps]).all():
        raise ValueError("Duration and fps must be positive and finite")
    model, cadence, checkpoint = load_checkpoint(args.checkpoint)
    runtime = OscillatorRuntime(
        model, checkpoint["tau_z_s"], cadence, cadence_rate_limit=args.cadence_rate_limit
    )
    timings = None
    if args.step_timings:
        payload = json.loads(args.step_timings.read_text())
        timings = StepTimings(tuple(payload["times"]), payload.get("first_foot", "left"))
        timings.sample(args.start_time + args.duration)
    commands = [{"time": args.start_time, "context": args.context}]
    if args.commands:
        commands = json.loads(args.commands.read_text())
        times = np.array([command["time"] for command in commands], dtype=float)
        if (
            not len(times)
            or not np.isfinite(times).all()
            or np.any(np.diff(times) <= 0)
            or times[0] > args.start_time
        ):
            raise ValueError("Command times must increase and cover the rollout start")
        for command in commands:
            runtime._context(command["context"])
    command_times = [command["time"] for command in commands]
    index = int(np.searchsorted(command_times, args.start_time, side="right")) - 1
    state = runtime.initial_state(
        commands[index]["context"], time=args.start_time, phase=args.phase, timings=timings
    )
    rows = []
    for _ in range(int(np.floor(args.duration * args.fps))):
        end_time = state.time + 1 / args.fps
        # Split integration at command boundaries, then sample on the requested frame grid.
        while index + 1 < len(commands) and command_times[index + 1] < end_time - 1e-10:
            boundary = float(command_times[index + 1])
            if boundary > state.time:
                runtime.advance(
                    state, commands[index]["context"], boundary - state.time, timings=timings
                )
            index += 1
        reference = runtime.advance(
            state, commands[index]["context"], end_time - state.time, timings=timings
        )
        rows.append((state.clone(), reference))
        if index + 1 < len(commands) and command_times[index + 1] <= state.time + 1e-10:
            index += 1
    if not rows:
        raise ValueError("Duration is shorter than one output frame")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        time=[s.time for s, _ in rows],
        phase=[s.phase for s, _ in rows],
        nominal_hz=[s.nominal_hz for s, _ in rows],
        z=np.stack([s.z.numpy() for s, _ in rows]),
        coordinates=np.stack([r.coordinates for _, r in rows]),
        coordinate_velocity=np.stack([r.coordinate_velocity for _, r in rows]),
        dof_pos=np.stack([r.coordinates[:22] for _, r in rows]),
        root_pos=np.stack([r.root_pos for _, r in rows]),
        root_rot_xyzw=np.stack([r.root_rot_xyzw for _, r in rows]),
        root_linear_velocity=np.stack([r.root_linear_velocity for _, r in rows]),
        root_angular_velocity=np.stack([r.root_angular_velocity for _, r in rows]),
        path_pos=np.stack([s.path_pos for s, _ in rows]),
        path_yaw=[s.path_yaw for s, _ in rows],
        phase_rate=[r.phase_rate for _, r in rows],
        joint_names=checkpoint["joint_names"],
        tau_z_s=runtime.tau_z,
        phase_control="step_timings" if timings else "nominal_cadence",
    )
    print(f"Saved {len(rows)} free-rollout frames to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="Download the pinned source and inspect clip schema")
    inspect.add_argument("--parquet", type=Path)
    inspect.add_argument("--revision", default=DATASET_REVISION)
    prepare = sub.add_parser("prepare", help="Fit root paths, label cycles and save split data")
    prepare.add_argument("--parquet", type=Path)
    prepare.add_argument("--revision", default=DATASET_REVISION)
    prepare.add_argument("--output", type=Path, default=Path("data/prepared"))
    prepare.add_argument("--splits", type=Path, help="JSON mapping train/dev/test to clip names")
    prepare.add_argument("--path-sigma", type=float, default=0.6)
    prepare.add_argument("--contact-height", type=float, default=0.045)
    prepare.add_argument("--contact-speed", type=float, default=0.6)
    prepare.add_argument("--unequal-halves", action="store_true")
    fit = sub.add_parser("train", help="Fit decoder; select checkpoint and smoothing on dev")
    fit.add_argument("--data", type=Path, default=Path("data/prepared"))
    fit.add_argument("--output", type=Path, default=Path("runs/baseline"))
    fit.add_argument("--epochs", type=int, default=300)
    fit.add_argument("--harmonics", type=int, default=3)
    fit.add_argument("--batch-size", type=int, default=512)
    fit.add_argument("--learning-rate", type=float, default=0.003)
    fit.add_argument("--seed", type=int, default=7)
    fit.add_argument("--device", choices=["cpu", "mps", "cuda"], default="cpu")
    fit.add_argument("--cadence", action="store_true", help="Fit the optional 81-parameter network")
    assess = sub.add_parser("evaluate", help="Re-evaluate a compatible prepared dataset")
    assess.add_argument("--data", type=Path, default=Path("data/prepared"))
    assess.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/model.pt"))
    assess.add_argument("--split", choices=["train", "dev", "test"], default="test")
    recon = sub.add_parser("reconstruct", help="Export observed-phase reconstruction")
    recon.add_argument("--data", type=Path, default=Path("data/prepared"))
    recon.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/model.pt"))
    recon.add_argument("--clip", default="walk3_subject2_04")
    recon.add_argument("--output", type=Path, default=Path("runs/baseline/reconstruction.npz"))
    recon.add_argument("--plot", type=Path, default=Path("runs/baseline/reconstruction.png"))
    free = sub.add_parser("rollout", help="Export a runtime rollout, optionally with transitions")
    free.add_argument("--checkpoint", type=Path, default=Path("runs/baseline/model.pt"))
    free.add_argument(
        "--context",
        type=float,
        nargs=4,
        default=[0.8, 0.0, 0.0, 0.9],
        metavar=("VX", "VY", "YAW_RATE", "CYCLE_PERIOD"),
    )
    free.add_argument("--commands", type=Path, help="JSON [{time, context}] command changes")
    free.add_argument("--step-timings", type=Path, help="JSON {times, first_foot} strike schedule")
    free.add_argument("--duration", type=float, default=5.0)
    free.add_argument("--fps", type=float, default=60.0)
    free.add_argument("--phase", type=float, default=0.0)
    free.add_argument("--start-time", type=float, default=0.0)
    free.add_argument("--cadence-rate-limit", type=float, default=1.0, help="cycle Hz per second")
    free.add_argument("--output", type=Path, default=Path("runs/baseline/rollout.npz"))
    args = parser.parse_args()
    if args.command == "inspect":
        clips = load_clips(args.parquet or download_dataset(args.revision))
        print(
            json.dumps(
                {
                    "revision": "local" if args.parquet else args.revision,
                    "joint_names": clips[0].joint_names,
                    "clips": [
                        {
                            "name": c.name,
                            "frames": len(c.time),
                            "fps": c.fps,
                            "duration_s": len(c.time) / c.fps,
                        }
                        for c in clips
                    ],
                },
                indent=2,
            )
        )
    elif args.command == "prepare":
        metadata = prepare_dataset(
            args.parquet or download_dataset(args.revision),
            args.output,
            PreprocessConfig(
                path_sigma_s=args.path_sigma,
                contact_height_m=args.contact_height,
                contact_speed_m_s=args.contact_speed,
                unequal_halves=args.unequal_halves,
            ),
            splits=None if args.splits is None else json.loads(args.splits.read_text()),
            revision="local" if args.parquet else args.revision,
        )
        for clip in metadata["clips"]:
            print(
                f"{clip['name']}: {len(clip['cycles'])} cycles, "
                f"{clip['valid_frames']}/{clip['frames']} valid frames; {clip['phase_source']}"
            )
        print(f"Saved prepared data to {args.output}")
    elif args.command == "train":
        report = train(
            args.data,
            args.output,
            TrainConfig(
                epochs=args.epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                seed=args.seed,
                device=args.device,
                cadence=args.cadence,
                harmonics=args.harmonics,
            ),
        )
        print(
            f"Saved {report['decoder_parameters']} decoder parameters to {args.output}; "
            f"tau_z={report['smoothing']['tau_z_s']} s"
        )
    elif args.command == "evaluate":
        model, cadence, checkpoint = load_checkpoint(args.checkpoint)
        metadata, clips = read_prepared(args.data)
        compatible_metadata(metadata, checkpoint)
        print(json.dumps(evaluate(model, clips, metadata["splits"][args.split], cadence), indent=2))
    elif args.command == "reconstruct":
        reconstruct(args)
    elif args.command == "rollout":
        rollout(args)


if __name__ == "__main__":
    main()
