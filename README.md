# Motion oscillator

A small PyTorch project for learning a shared whole-body oscillator from
[LAFAN locomotion retargeted to the Booster K1](https://huggingface.co/datasets/whirlwind-ams/lafan_locomotion_k1).
It implements the first candidate architecture: a phase clock separated from a conditional
harmonic decoder, with four shared mixture states and 28 output coordinates.

## Quick start

Python 3.12 and [uv](https://docs.astral.sh/uv/) are used for the checked baseline. The project
supports Python 3.11–3.13 and CPU training; no simulator or GPU is required.

```bash
uv sync
uv run oscillator inspect
uv run oscillator prepare
uv run oscillator train
uv run oscillator evaluate
uv run oscillator reconstruct
uv run oscillator rollout --commands examples/commands.json
uv run pytest
```

## Browser playback viewer

The static viewer in `web/` includes the fitted baseline, all seven source clips, K1 visual
meshes and locally bundled rendering dependencies. Start it without rebuilding assets:

```bash
python3 -m http.server 8000 --bind 127.0.0.1 --directory web
```

Open [localhost:8000](http://localhost:8000). **Compare clips** plays the recorded and decoded
K1 poses side by side with source contact estimates, root paths, a phase indicator and joint
angle traces. The robots use fixed display offsets; reconstruction errors are preserved.
Unlabelled source frames show no learned pose. **Generate motion** runs the exported decoder
in the browser with live forward/lateral velocity, turning and cycle-period controls, shared
mixture smoothing and a transition preset. Both modes support playback speed, pause/restart,
time scrubbing, orbit/zoom, camera following and JSON motion export. Exports preserve world
coordinates and omit the display offsets. Generated exports run a separate preview state,
so exporting does not advance live playback.

This is kinematic pose playback. Contact correction and a dynamics controller are not
applied. Browser model evaluation and runtime transitions are tested against the actual
PyTorch checkpoint; the K1 hierarchy is reconstructed from the pinned upstream MJCF file.

To refresh the viewer after training a new checkpoint:

```bash
npm ci
uv run python scripts/build_viewer.py --checkpoint runs/baseline/model.pt
npm test
```

This exports model parameters, normalisation buffers, little-endian float32 clip files,
split/accuracy metadata and cross-language validation fixtures. The first build downloads
the pinned upstream K1 visual assets. Subsequent builds use the local copies. `npm ci`
is only needed to rebuild the vendored Three.js dependencies. Playback itself makes no
requests to external CDNs or inference services.

The public viewer is hosted at
[oliverobst.github.io/oscillator](https://oliverobst.github.io/oscillator/), with source at
[OliverObst/oscillator](https://github.com/OliverObst/oscillator).
The deployment workflow at `.github/workflows/pages.yml` publishes the static `web/`
directory whenever its files change on `main`. Local playback uses the same files.

`prepare` downloads the 3.2 MB Parquet file through the Hugging Face cache. The default source
revision is pinned to `8cb1332286a7d28d3f9ffc4c27bb2de3b861cbae`. The source hash, configuration,
joint order, phase diagnostics and splits are saved in `data/prepared/metadata.json`.
Use `--parquet FILE` to work offline with a local copy, or `--revision SHA` to intentionally
change the source. Prepared data and run artefacts are ignored by Git; `uv.lock` is tracked.

Training saves `runs/baseline/model.pt` and `report.json`. Reconstruction saves an NPZ archive
and a PNG plot. All output locations can be changed with `--output`; use `--plot` to change
the reconstruction plot destination. Run `uv run oscillator COMMAND --help` for options.

## Decoder

For context `ζ = [mean_vx, mean_vy, mean_yaw_rate, cycle_period]`, the network is
`4 → 32 → 32 → 4`, with tanh hidden layers and a linear mixture output. It mixes four
variation waveforms around a mean:

```text
z = context_net(normalise(ζ))
y(φ, ζ) = B₀(φ) + Σᵣ zᵣ Bᵣ(φ), r = 1,…,4
Bᵣ(φ) = aᵣ₀ + Σₖ [aᵣₖ cos(2πkφ) + bᵣₖ sin(2πkφ)], k = 1,…,3
```

Each waveform contains all 28 coordinates. The coefficient tensor is `(5, 28, 7)`, ordered
`[DC, cos₁, sin₁, cos₂, sin₂, cos₃, sin₃]`. There are 980 waveform coefficients and 1,348
context-network parameters: **2,328 decoder parameters**. Input and target normalisation
statistics are buffers fitted on training frames only. Decoding returns physical units.
The loss is normalised coordinate MSE with mixture-output and harmonic-energy penalties
(harmonic weights proportional to `k⁴`), plus small AdamW weight decay.

The optional `3 → 16 → 1` cadence network adds **81 parameters**. Its sigmoid output is
bounded to 0.5–3.0 **cycles per second**, and it is fitted to reciprocal observed cycle
periods, with one example per cycle. It receives the three velocity commands rather than
the requested period:

```bash
uv run oscillator train --cadence --output runs/cadence
uv run oscillator rollout --checkpoint runs/cadence/model.pt
```

Without that network, the nominal clock uses `1 / cycle_period`. In either case the runtime
rate-limits nominal cadence and gives the decoder the actual clock period. Supplied step
timings replace cadence control completely, including bypassing the cadence network.
Preprocessing, path integration, contact supervision and downstream kinematics are excluded
from the parameter count.

## Coordinates and root path

The output order is all 22 source joint angles, followed by `[dx, dy, height, rx, ry, rz]`.
Angles and rotation vectors use radians; translations use metres. The source quaternions
are `xyzw`, and the vertical axis is world `z`. Planar velocity commands are expressed in
the fitted path's heading frame.

For each clip, preprocessing fits a slowly progressing planar path: a Gaussian low-pass
filter with a fixed 0.6 s standard deviation is applied to root `x/y` and unwrapped root
heading. Path height is zero, so root height remains an output coordinate. This is an
offline, non-causal fit with a fixed bandwidth; the path is saved explicitly. There is no
per-frame optimisation to align predictions to the target.

For fitted path position `p_path` and heading rotation `R_path`, root residuals are:

```text
d = R_path⁻¹ (p_root − p_path)
r = Log(R_path⁻¹ R_root)
p_root = p_path + R_path d
R_root = R_path Exp(r)
```

This avoids absolute yaw wraps. Exact preprocessing round-trip errors are recorded before
fitting the decoder. Root reconstruction uses the saved path. Free runtime rollout instead
integrates commanded planar motion with exact constant-command SE(2) integration; it never
realigns the output to observed frames. Root world velocities include path motion and the
rotation-vector Jacobian, as well as decoder derivatives.

## Phase supervision and splits

The source has seven complete clips at 30 Hz, all in its `train` split. This project makes
its own **whole-clip** split before fitting normalisation or model parameters:

| Split | Clips |
| --- | --- |
| Train | `run2_subject1_05`, `walk3_subject2_01`, `walk3_subject2_02` |
| Development | `run2_subject1_06`, `walk3_subject2_03` |
| Test | `run2_subject1_07`, `walk3_subject2_04` |

These are clip-held-out experiments, with the same subjects appearing across splits; they
do not establish generalisation to unseen subjects. Provide `prepare --splits FILE` with a
JSON mapping `train`, `dev` and `test` to non-overlapping clip lists to change the protocol.
Every source clip must be assigned exactly once.

Foot-body positions are transformed from the trunk frame to world space. Provisional
contact labels use height within 4.5 cm of each foot's tenth-percentile height and speed
below 0.6 m/s. Left contact onsets anchor phase zero; successive plausible left strikes
form full cycles. Periods outside 0.35–2.0 s, gaps and incomplete boundary cycles are
excluded rather than interpolated through. By default phase progresses uniformly within
each labelled cycle. Context is the cycle average of fitted-path velocities and the full
cycle period, so these are offline labels, not a causal command estimator.

When too few contact cycles are detected, left-foot height minima provide an explicitly
recorded fallback. In the pinned source this affects `walk3_subject2_03`: only three cycles
and 131/316 frames are retained. Phase estimates and contact thresholds need further
inspection before drawing strong conclusions about the architecture. They are kinematic
heuristics, not measured ground-reaction contacts.

Unequal half-cycle timing is an explicit alternative:

```bash
uv run oscillator prepare --unequal-halves --output data/unequal
uv run oscillator train --data data/unequal --output runs/unequal
```

This requires exactly one usable right strike inside each left-to-left cycle, places it at
phase 0.5, and records each half's phase rate. Cycles lacking that supervision are excluded.
The pinned dataset's current contact heuristics do not yield three such cycles in every
clip, so this preparation command currently stops with a diagnostic identifying the clip.
The extension is implemented and tested on known timing labels, but needs improved contact
supervision for a full dataset fit. Changing preprocessing requires a new fit.

## Changing commands and previews

`OscillatorRuntime` holds time, phase, nominal cadence, four mixture values, planar path
position and unwrapped heading. `RuntimeState.clone()` copies every mutable value, so a
preview can advance independently. The mixture update is the exact exponential solution
for a held target, `z_next = z_target + (z − z_target) exp(−dt / τ_z)`; with a changing clock
period, the target is evaluated at each runtime step.

The fixed `τ_z` is selected from 0.05, 0.1, 0.2, 0.4 and 0.8 s using natural development
cycle-context transitions and one-cycle reconstruction windows following each change.
The baseline selects 0.1 s. This is a development proxy; dedicated command-transition data
would be needed to establish performance on arbitrary manoeuvres. Test clips are evaluated
only after checkpoint and smoothing selection.

Reference rates include both `∂y/∂φ · φ_dot` and `Σᵣ Bᵣ · z_dotᵣ`. Smoothing happens before
the optional `adjustment(reference, state)` hook. The hook is the integration point for
contact-constrained reference generation/IK and must supply consistent derivatives. This
baseline has no foot-planting solver, joint-limit enforcement or dynamic robot controller.
It applies no independent joint-target filtering after the hook.

The provided command file changes velocity, turning and period at specified times:

```bash
uv run oscillator rollout --commands examples/commands.json --output runs/baseline/transitions.npz
uv run oscillator rollout --step-timings examples/step_timings.json --output runs/baseline/timed.npz
```

Step-timing JSON contains `times` and `first_foot`; times are absolute seconds and alternate
between left and right strikes. The first strike anchors phase zero for left, or 0.5 for
right. Unequal halves are linearly interpolated when a schedule is supplied. The schedule
must cover the full rollout; exhaustion raises an error. The caller/contact supervisor is
responsible for providing these timings. There is no automatic online contact correction.

## Initial baseline and evaluation

With seed 7, 300 epochs, default preprocessing and CPU training, development selection
retained epoch 103. The held-out results are:

| Test clip | Reconstruction joint RMSE (rad) | Uniform rollout joint RMSE (rad) | Rollout phase RMSE (cycles) |
| --- | ---: | ---: | ---: |
| `run2_subject1_07` | 0.305 | 0.326 | 0.298 |
| `walk3_subject2_04` | 0.244 | 0.318 | 0.283 |

Reconstruction uses observed phase, cycle context and the saved fitted path; it measures
decoder approximation, not autonomous rollout. Uniform rollout resets phase once at the
first valid frame, holds an oracle clip-average context and constant nominal cadence, and
integrates that constant command's root path from the first fitted path pose. The optional
cadence network replaces only the frequency estimate. Neither protocol establishes fully
autonomous command inference. Reports contain joint and root-residual errors, world root
position/orientation errors and circular phase drift; excluded frames are counted explicitly.

The long free rollouts drift, and held-out reconstruction errors remain substantial. The
next modelling decisions should be informed by phase-label inspection, non-periodic motion
and controlled transition data. The pipeline makes those limitations measurable.

The [controllable upstream demo](https://intelligentroboticslab.github.io/booster_mjlab/)
is a potential complementary source: it exposes its simulator as `window.k1demo.view.sim`.
A recorder could sample joints, root poses, commands and physical contacts at simulation
steps, splitting clips on resets. Its physics timestep is 5 ms and its policy interval is
20 ms. Forward commands span −1.0 to 1.5 m/s, lateral commands −1.0 to 1.0 m/s, and yaw
commands −1.5 to 1.5 rad/s. Cadence emerges from the policy. Such recordings would be
synthetic policy-generated motion and should retain separate provenance and evaluation
splits. A recorder is not included in this first implementation.

## Code and checks

- `data.py`: pinned download and schema validation.
- `preprocess.py`: root path, contacts, cycle labels and split metadata.
- `model.py`: harmonic decoder, analytic reference derivatives and optional cadence.
- `runtime.py`: cloneable shared state, timing precedence, smoothing and path integration.
- `train.py`: fitting, development selection, checkpoints and evaluation.
- `cli.py`: inspection, preparation, training, reconstruction and rollout commands.

```bash
uv run ruff check .
uv run ruff format --check .
uv run pytest
```

Tests cover parameter counts, periodicity, derivatives against finite differences, root
round trips across yaw wraps, path preservation, phase masking, split leakage, preview
isolation, cadence precedence, schedule exhaustion and train-only checkpoint selection.
The repository licence applies to project code; source motion data remain subject to their
own upstream terms. The source schema's trunk-local body-position convention was checked
against [booster_mjlab's motion loader](https://github.com/IntelligentRoboticsLab/booster_mjlab/blob/main/src/booster_mjlab/motion/motion_data.py).
