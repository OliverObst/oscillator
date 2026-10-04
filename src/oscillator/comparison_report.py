"""Write a portable, reproducible account of the frozen decoder comparison."""

from html import escape
from pathlib import Path

import numpy as np


def mean_test_metric(candidates, key):
    return np.mean([c["aggregate"]["test"][key] for c in candidates])


def write_comparison_pages(comparison, root: Path):
    rows, markdown = [], []
    for group in comparison["groups"].values():
        candidates = group["candidates"]
        low, high = group["test_joint_rmse_range_rad"]
        values = [
            group["label"],
            f"{candidates[0]['parameters']:,}",
            f"{np.mean([c['best_dev_mse'] for c in candidates]):.3f}",
            f"{group['test_joint_rmse_mean_rad']:.3f} ({low:.3f}–{high:.3f})",
            f"{mean_test_metric(candidates, 'foot_position_rmse_m'):.3f}",
            f"{mean_test_metric(candidates, 'learned_contact_horizontal_speed_rms_m_s'):.3f}",
        ]
        rows.append("<tr>" + "".join(f"<td>{escape(v)}</td>" for v in values) + "</tr>")
        markdown.append("| " + " | ".join(values) + " |")
    protocol = comparison["protocol"]
    settings = comparison["groups"]["rff"]["settings"]
    settings_text = (
        f"40 random projections (80 features), phase bandwidth {settings['phase_bandwidth']}, "
        f"context bandwidth {settings['context_bandwidth']}, ridge {settings['ridge']:g}."
    )
    interpretation = (
        "RFF improves mean held-out joint and foot-position errors, but the six-harmonic "
        "model has the lowest development normalised MSE. This small comparison does "
        "not establish RFF as the overall winner. Substantial contact sliding remains."
    )
    measurement = (
        "Test numbers are means across seeds 7, 17 and 27; parentheses give the seed range "
        "for joint RMSE. Within each seed, clips are pooled by valid frame count before "
        "computing RMSE. Foot position error is world-space Euclidean RMSE over both feet, "
        "using the saved fitted planar path with no pose alignment. Contact speed is RMS "
        "horizontal foot speed over consecutive frames labelled as contact at both ends. "
        "Source contact speed is 0.238 m/s on these test clips; labels are provisional. "
        "These are observed-phase reconstructions, not free-rollout accuracy."
    )
    runtime = (
        "RFF uses [cos(2πφ), sin(2πφ), normalised context] with fixed Gaussian projections, "
        "paired sine/cosine features and a learned 28-output ridge readout. Bandwidths scale "
        "the projection entries directly, in radians per input unit. No extra 2π factor "
        "is applied to random projections. Four normalised context values are smoothed "
        "before decoding; both phase advance and input-state changes contribute to velocity. "
        "The harmonic variants smooth their four mixture values. All variants use one "
        "rate-limited phase clock and the same explicit root-path integration. None applies IK "
        "or foot-contact constraints."
    )
    fitting = (
        "Both harmonic variants use 300 epochs and the baseline optimisation/regularisation "
        "settings; best epochs are selected on development error. RFF uses a fixed grid: "
        "phase bandwidths {1, 2, 4}, context bandwidths {0.25, 0.5, 1}, ridge penalties "
        "{1e−7, 1e−6, 1e−5, 1e−4}. Choose the grid cell by mean development normalised MSE "
        "across the same three seeds. Ridge minimises mean squared error over frames and "
        "28 coordinates plus λ times the sum of squared readout weights; bias is unpenalised. "
        "Fixed random projections (240 stored scalars), normalisation and preprocessing are "
        "excluded from learned parameter counts. Smoothing is selected separately per model "
        "using natural development transitions."
    )
    viewer = (
        "The viewer retains the original three-harmonic baseline (seed 7, τ=0.1 s). "
        "The six-harmonic and RFF representatives use development-selected seeds 27 and 7, "
        "with τ=0.05 and 0.1 s respectively. Clip readouts show that individual model's error; "
        "this table shows means over all three seeds."
    )
    title = "Periodic random features: the first comparison"
    paper_url = (
        "https://papers.nips.cc/paper_files/paper/2007/hash/"
        "013a006f03dbc5392effeb8f18fda755-Abstract.html"
    )
    document_url = (
        "https://github.com/OliverObst/oscillator/blob/main/experiments/decoder-comparison.md"
    )
    html = f"""<!doctype html>
<html lang="en-AU"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Decoder comparison · Motion oscillator</title>
<link rel="stylesheet" href="./style.css"><link rel="icon" href="./favicon.svg"></head>
<body><header class="site-header"><a class="brand" href="./">
<img src="./favicon.svg" width="32" height="32" alt="">Motion oscillator</a>
<a href="./">Back to viewer ↗</a></header><main class="notes">
<p class="eyebrow">DEVELOPMENT-SELECTED EXPERIMENT</p><h1>{title}</h1><p>{interpretation}</p>
<div class="results-table"><table><thead><tr><th>Decoder</th><th>Parameters</th><th>Dev MSE</th>
<th>Test joint RMSE (rad)</th><th>Foot RMSE (m)</th><th>Contact speed (m/s)</th>
</tr></thead><tbody>{"".join(rows)}</tbody></table></div>
<p>{measurement}</p><h2>How models were selected</h2><p>{protocol}</p><p>{fitting}</p>
<h2>RFF model and runtime</h2><p>{settings_text}</p><p>{runtime}</p>
<h2>Viewer models</h2><p>{viewer}</p>
<p><a href="./assets/comparison.json">Full results JSON</a> ·
<a href="{document_url}">Reproduce this experiment</a></p>
<p>RFF background: <a href="{paper_url}">Rahimi &amp; Recht (2007)</a>.</p>
<a class="back" href="./">← Back to playback</a></main></body></html>
"""
    (root / "web/comparison.html").write_text(html)
    directory = root / "experiments"
    directory.mkdir(exist_ok=True)
    table_header = (
        "| Decoder | Learned parameters | Dev normalised MSE | Test joint RMSE (rad; range) "
        "| Foot RMSE (m) | Contact speed (m/s) |\n"
        "|---|---:|---:|---:|---:|---:|"
    )
    text = f"# {title}\n\n{interpretation}\n\n{table_header}\n" + "\n".join(markdown)
    text += (
        f"\n\n{measurement}\n\n## Selection\n\n{protocol}\n\n{fitting}"
        f"\n\n## Model\n\n{settings_text}\n\n{runtime}\n\n{viewer}\n"
    )
    text += """\n## Reproduce

```bash
uv sync
uv run oscillator prepare
uv run python scripts/compare_decoders.py
npm ci
uv run python scripts/build_viewer.py --comparison runs/comparison/comparison.json
npm test
npm run serve
```

`prepare` uses the pinned dataset revision and default clip split. Experiment checkpoints,
training curves, the full grid and per-seed reports are in `runs/comparison/` (ignored by Git).
`selection.json` is written before test evaluation. To resume evaluation after an interruption,
use `uv run python scripts/compare_decoders.py --evaluate-selection`.
The static viewer exports selected parameters and the portable complete report in
`web/assets/comparison.json`. The original baseline is reproduced by seed 7 with three harmonics.
There is no new data from the upstream locomotion demo in this experiment.
"""
    (directory / "decoder-comparison.md").write_text(text)
