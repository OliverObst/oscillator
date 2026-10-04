# Periodic random features: the first comparison

RFF improves mean held-out joint and foot-position errors, but the six-harmonic model has the lowest development normalised MSE. This small comparison does not establish RFF as the overall winner. Substantial contact sliding remains.

| Decoder | Learned parameters | Dev normalised MSE | Test joint RMSE (rad; range) | Foot RMSE (m) | Contact speed (m/s) |
|---|---:|---:|---:|---:|---:|
| 3 harmonics | 2,328 | 0.999 | 0.280 (0.279–0.280) | 0.272 | 0.920 |
| 6 harmonics | 3,168 | 0.987 | 0.276 (0.268–0.282) | 0.241 | 0.827 |
| Periodic RFF | 2,268 | 1.043 | 0.264 (0.262–0.267) | 0.235 | 0.749 |

Test numbers are means across seeds 7, 17 and 27; parentheses give the seed range for joint RMSE. Within each seed, clips are pooled by valid frame count before computing RMSE. Foot position error is world-space Euclidean RMSE over both feet, using the saved fitted planar path with no pose alignment. Contact speed is RMS horizontal foot speed over consecutive frames labelled as contact at both ends. Source contact speed is 0.238 m/s on these test clips; labels are provisional. These are observed-phase reconstructions, not free-rollout accuracy.

## Selection

Train-only normalisation and fitting; development-only bandwidth, ridge, epoch, smoothing and representative seed selection. Test evaluated afterwards. Three clips train, two develop, two test; subjects overlap. Test clips were previously inspected for the original baseline. Seed ranges measure model randomness, not independent datasets or confidence intervals.

Both harmonic variants use 300 epochs and the baseline optimisation/regularisation settings; best epochs are selected on development error. RFF uses a fixed grid: phase bandwidths {1, 2, 4}, context bandwidths {0.25, 0.5, 1}, ridge penalties {1e−7, 1e−6, 1e−5, 1e−4}. Choose the grid cell by mean development normalised MSE across the same three seeds. Ridge minimises mean squared error over frames and 28 coordinates plus λ times the sum of squared readout weights; bias is unpenalised. Fixed random projections (240 stored scalars), normalisation and preprocessing are excluded from learned parameter counts. Smoothing is selected separately per model using natural development transitions.

## Model

40 random projections (80 features), phase bandwidth 1.0, context bandwidth 0.5, ridge 0.0001.

RFF uses [cos(2πφ), sin(2πφ), normalised context] with fixed Gaussian projections, paired sine/cosine features and a learned 28-output ridge readout. Bandwidths scale the projection entries directly, in radians per input unit. No extra 2π factor is applied to random projections. Four normalised context values are smoothed before decoding; both phase advance and input-state changes contribute to velocity. The harmonic variants smooth their four mixture values. All variants use one rate-limited phase clock and the same explicit root-path integration. None applies IK or foot-contact constraints.

The viewer retains the original three-harmonic baseline (seed 7, τ=0.1 s). The six-harmonic and RFF representatives use development-selected seeds 27 and 7, with τ=0.05 and 0.1 s respectively. Clip readouts show that individual model's error; this table shows means over all three seeds.

## Reproduce

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
