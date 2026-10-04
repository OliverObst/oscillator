"""Development-selected harmonic/RFF comparison; test evaluation follows all selection."""

import json
from dataclasses import asdict
from itertools import product
from pathlib import Path

import numpy as np
import torch

from .model import RFFDecoder, decoder_spec
from .preprocess import read_prepared
from .train import TrainConfig, evaluate, frame_tensors, load_checkpoint, select_smoothing, train


@torch.no_grad()
def normalised_mse(model, frames):
    target = (frames["target"] - model.target_mean) / model.target_scale
    return float(
        (model(frames["phase"], frames["context"], normalised=True) - target).square().mean()
    )


def aggregate(metrics):
    count = sum(c["frames"] for c in metrics.values())
    keys = (
        "joint_rmse_rad",
        "root_residual_translation_rmse_m",
        "world_root_position_rmse_m",
        "world_root_orientation_rmse_rad",
        "foot_position_rmse_m",
    )
    result = {
        key: float(
            np.sqrt(
                sum(c["frames"] * c["reconstruction"][key] ** 2 for c in metrics.values()) / count
            )
        )
        for key in keys
    }
    contact_count = sum(c["reconstruction"]["contact_samples"] for c in metrics.values())
    for kind in ("learned", "source"):
        key = f"{kind}_contact_horizontal_speed_rms_m_s"
        result[key] = (
            float(
                np.sqrt(
                    sum(
                        c["reconstruction"]["contact_samples"]
                        * (c["reconstruction"][key] or 0) ** 2
                        for c in metrics.values()
                    )
                    / contact_count
                )
            )
            if contact_count
            else None
        )
    return result


def compare_decoders(
    prepared: Path,
    output: Path,
    skeleton_path: Path,
    epochs=300,
    seeds=(7, 17, 27),
    phase_bandwidths=(1.0, 2.0, 4.0),
    context_bandwidths=(0.25, 0.5, 1.0),
    ridges=(1e-7, 1e-6, 1e-5, 1e-4),
):
    """Fixed grid shared across random seeds; select by mean development normalised MSE.

    Harmonic regularisation/optimisation is held at the published baseline settings.
    Test clips have previously been viewed for the original baseline, so this is a
    previously inspected hold-out, not a new blind test. This run never tunes on it.
    """
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Supply distinct non-empty random seeds")
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    metadata, clips = read_prepared(prepared)
    splits = metadata["splits"]
    training = frame_tensors(clips, splits["train"], "cpu")
    dev = frame_tensors(clips, splits["dev"], "cpu")
    skeleton = json.loads(skeleton_path.read_text())
    if skeleton["joint_names"] != metadata["joint_names"]:
        raise ValueError("Kinematic and dataset joint orders disagree")
    groups = {}
    for harmonics in (3, 6):
        name, candidates = f"harmonic{harmonics}", []
        for seed in seeds:
            directory = output / name / f"seed{seed}"
            report = train(
                prepared,
                directory,
                TrainConfig(epochs=epochs, seed=seed, harmonics=harmonics),
                evaluate_test=False,
            )
            candidates.append(
                {
                    "seed": seed,
                    "best_dev_mse": report["best_dev_mse"],
                    "best_epoch": report["best_epoch"],
                    "checkpoint": str((directory / "model.pt").resolve()),
                }
            )
        groups[name] = {"label": f"{harmonics} harmonics", "candidates": candidates}
    grid = []
    rff_models = {}
    for phase_bw, context_bw in product(phase_bandwidths, context_bandwidths):
        models = []
        for seed in seeds:
            model = RFFDecoder(40, phase_bw, context_bw, seed)
            model.set_normalisation(training["context"], training["target"])
            models.append(model)
        for ridge in ridges:
            scores, fitted = [], []
            for model in models:
                model.fit_ridge(training["phase"], training["context"], training["target"], ridge)
                scores.append(normalised_mse(model, dev))
                fitted.append({k: v.clone() for k, v in model.state_dict().items()})
            settings = {
                "phase_bandwidth": phase_bw,
                "context_bandwidth": context_bw,
                "ridge": ridge,
            }
            grid.append(
                {**settings, "dev_mse_by_seed": scores, "mean_dev_mse": float(np.mean(scores))}
            )
            rff_models[(phase_bw, context_bw, ridge)] = fitted
    best = min(grid, key=lambda row: row["mean_dev_mse"])
    settings = {key: best[key] for key in ("phase_bandwidth", "context_bandwidth", "ridge")}
    print(f"Selected RFF settings on development clips: {settings}", flush=True)
    candidates = []
    states = rff_models[
        (settings["phase_bandwidth"], settings["context_bandwidth"], settings["ridge"])
    ]
    for seed, state, score in zip(seeds, states, best["dev_mse_by_seed"], strict=True):
        model = RFFDecoder(40, settings["phase_bandwidth"], settings["context_bandwidth"], seed)
        model.load_state_dict(state)
        tau, smoothing = select_smoothing(model, clips, splits["dev"], TrainConfig().tau_candidates)
        directory = output / "rff" / f"seed{seed}"
        directory.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "schema_version": 1,
            "decoder_spec": decoder_spec(model),
            "decoder_state": state,
            "cadence_state": None,
            "cadence_bounds": None,
            "tau_z_s": tau,
            "joint_names": metadata["joint_names"],
            "preprocessing": metadata,
            "train_config": {**settings, "seed": seed, "fit": "ridge", "projections": 40},
        }
        torch.save(checkpoint, directory / "model.pt")
        report = {
            "decoder_parameters": model.parameter_count,
            "best_dev_mse": score,
            "smoothing": smoothing,
            "metrics": {},
            "cadence_parameters": 0,
        }
        (directory / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        candidates.append(
            {
                "seed": seed,
                "best_dev_mse": score,
                "checkpoint": str((directory / "model.pt").resolve()),
            }
        )
    groups["rff"] = {"label": "Periodic RFF", "settings": settings, "candidates": candidates}
    # Freeze all settings and representative seeds before accessing test targets.
    for group in groups.values():
        group["selected_seed"] = min(group["candidates"], key=lambda c: c["best_dev_mse"])["seed"]
    selection = {
        "seeds": list(seeds),
        "groups": groups,
        "rff_grid": grid,
        "protocol": "Train-only normalisation and fitting; development-only bandwidth, ridge, "
        "epoch, smoothing and representative seed selection. Test evaluated afterwards. "
        "Three clips train, two develop, two test; subjects overlap. Test clips were "
        "previously inspected for the original baseline. Seed ranges measure model "
        "randomness, not independent datasets or confidence intervals.",
        "harmonic_train_config": asdict(TrainConfig(epochs=epochs)),
        "source_sha256": metadata["source_sha256"],
        "splits": splits,
    }
    (output / "selection.json").write_text(json.dumps(selection, indent=2) + "\n")
    return evaluate_selection(prepared, output, skeleton_path)


def evaluate_selection(prepared: Path, output: Path, skeleton_path: Path):
    """Evaluate a saved frozen selection; permits recovery without re-fitting or tuning."""
    torch.set_num_threads(2)
    selection = json.loads((output / "selection.json").read_text())
    metadata, clips = read_prepared(prepared)
    splits = metadata["splits"]
    if metadata["source_sha256"] != selection["source_sha256"] or splits != selection["splits"]:
        raise ValueError("Frozen selection and prepared data disagree")
    skeleton = json.loads(skeleton_path.read_text())
    if skeleton["joint_names"] != metadata["joint_names"]:
        raise ValueError("Kinematic and dataset joint orders disagree")
    groups = selection["groups"]
    for group in groups.values():
        for candidate in group["candidates"]:
            checkpoint_path = Path(candidate["checkpoint"])
            model, cadence, _ = load_checkpoint(checkpoint_path)
            metrics = {
                split: evaluate(model, clips, names, cadence, skeleton=skeleton)
                for split, names in splits.items()
            }
            report_path = checkpoint_path.parent / "report.json"
            report = json.loads(report_path.read_text())
            report["metrics"] = metrics
            report["test_normalised_mse"] = normalised_mse(
                model, frame_tensors(clips, splits["test"], "cpu")
            )
            report_path.write_text(json.dumps(report, indent=2) + "\n")
            candidate.update(
                parameters=model.parameter_count,
                tau_z_s=report["smoothing"]["tau_z_s"],
                aggregate={split: aggregate(m) for split, m in metrics.items()},
                metrics=metrics,
            )
        chosen = next(c for c in group["candidates"] if c["seed"] == group["selected_seed"])
        group["selected_checkpoint"] = chosen["checkpoint"]
        group["test_joint_rmse_mean_rad"] = float(
            np.mean([c["aggregate"]["test"]["joint_rmse_rad"] for c in group["candidates"]])
        )
        group["test_joint_rmse_range_rad"] = [
            float(f([c["aggregate"]["test"]["joint_rmse_rad"] for c in group["candidates"]]))
            for f in (np.min, np.max)
        ]
    (output / "comparison.json").write_text(json.dumps(selection, indent=2) + "\n")
    print(
        json.dumps(
            {
                name: {
                    k: group[k]
                    for k in (
                        "selected_seed",
                        "test_joint_rmse_mean_rad",
                        "test_joint_rmse_range_rad",
                    )
                }
                for name, group in groups.items()
            },
            indent=2,
        )
    )
    return selection
