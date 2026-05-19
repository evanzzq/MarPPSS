# MarPPSS Workflow

This is a quick-start guide for the current config-driven workflow.

## 1. Define paths

Set these in your shell before running anything:

```powershell
$env:MARPPSS_DATA_DIR = "H:\My Drive\Research\MarPPSS"
$env:MARPPSS_CODE_DIR = "C:\Users\zzq\Documents\Research\MarsPPSS"
```

or

```bash
export MARPPSS_DATA_DIR="/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/MarPPSS"
export MARPPSS_CODE_DIR="/Users/evanzhang/Documents/Research/MarPPSS"
```

`MARPPSS_DATA_DIR` is the root directory that contains the `data/` and `run/` folders.

`MARPPSS_CODE_DIR` is the path to this repository.

On Windows `cmd.exe`, use `set MARPPSS_DATA_DIR=...` and `set MARPPSS_CODE_DIR=...` instead.
In PowerShell, replace `$MARPPSS_CODE_DIR` with `$env:MARPPSS_CODE_DIR` and `$MARPPSS_DATA_DIR` with `$env:MARPPSS_DATA_DIR` in the commands below.

## 2. Choose the config file

The example workflow config is:

```bash
$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml
```

In PowerShell, use:

```powershell
$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml
```

## 3. List available experiments

```bash
python -m marppss list "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

PowerShell:

```powershell
python -m marppss list "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml"
```

This prints both configured events and configured experiments.

## 4. Prep the event once

This reads the SAC files, applies filtering/windowing, builds the reference wavelet, and writes processed files into `data/`.

If you want no bandpass filtering, set `PPfreq: null` and `SSfreq: null` for that event in the YAML. You can also disable filtering for only one component by setting just that one to `null`.

Example:

```bash
python -m marppss prep "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --event S0976asdr_1883
```

PowerShell:

```powershell
python -m marppss prep "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" --event S0976asdr_1883
```

You only need to do this once for a given event/preprocessing setup.

## 5. Run one experiment

Example:

```bash
python -m marppss run "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --experiment s0976asdr_1883_gv_1to40_20
```

PowerShell:

```powershell
python -m marppss run "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" --experiment s0976asdr_1883_gv_1to40_20
```

Another example:

```bash
python -m marppss run "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --experiment s0976asdr_1883_pp_tt_gv_20_30
```

PowerShell:

```powershell
python -m marppss run "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" --experiment s0976asdr_1883_pp_tt_gv_20_30
```

## 6. Run multiple experiments in batch

Dry run first:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_ \
  --dry-run
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\run_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_gv_ `
  --dry-run
```

Run all `GV only` experiments:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\run_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_gv_
```

Run all `PP TT + GV` experiments:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_pp_tt_gv_
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\run_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_pp_tt_gv_
```

## 7. Plot one run

Example for a `GV only` run:

```bash
python -m marppss plot \
  "$MARPPSS_DATA_DIR/run/S0976asdr_1883_src_3.5_s_SS/run_gv_1to40_20" \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

PowerShell:

```powershell
python -m marppss plot `
  "$env:MARPPSS_DATA_DIR\run\S0976asdr_1883_src_3.5_s_SS\run_gv_1to40_20" `
  --reference-config "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml"
```

Example for a `PP TT + GV` run:

```bash
python -m marppss plot \
  "$MARPPSS_DATA_DIR/run/S0976asdr_1883_src_3.5_s_PP/run_pp_tt_gv_20_30" \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

PowerShell:

```powershell
python -m marppss plot `
  "$env:MARPPSS_DATA_DIR\run\S0976asdr_1883_src_3.5_s_PP\run_pp_tt_gv_20_30" `
  --reference-config "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml"
```

Plotting flow:

1. show likelihood diagnostics for all chains
2. choose how many top chains to keep
3. show the rest of the figures for that subset

## 8. Plot multiple runs in batch

Dry run first:

```bash
python "$MARPPSS_CODE_DIR/scripts/plot_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_ \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --dry-run
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\plot_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_gv_ `
  --reference-config "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --dry-run
```

Plot all `GV only` runs:

```bash
python "$MARPPSS_CODE_DIR/scripts/plot_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_ \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\plot_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_gv_ `
  --reference-config "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml"
```

Plot all `PP TT + GV` runs:

```bash
python "$MARPPSS_CODE_DIR/scripts/plot_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_pp_tt_gv_ \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

PowerShell:

```powershell
python "$env:MARPPSS_CODE_DIR\scripts\plot_batch.py" `
  "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml" `
  --prefix s0976asdr_1883_pp_tt_gv_ `
  --reference-config "$env:MARPPSS_CODE_DIR\configs\mars_workflow.example.yaml"
```

If you want to use the same number of top chains for every run, add:

```bash
--top-chains 6
```

If you omit `--top-chains`, batch plotting will stop at each run and let you choose interactively.

## 9. Notes

- `reference_model` in the YAML should be specified using discontinuity depths:

```yaml
reference_model:
  H: [11.0, 47.5]
  vp: [5.6, 5.6, 6.8]
  vs: [3.1, 3.1, 3.8]
```

- `H` is depth, not thickness.
- For travel-time experiments, the number of layers is now inferred automatically from the number of supplied travel-time picks.
- To invert for gradational layers, set `assumptions.top_layer_gradient` and give `aRange`. `sqrt` and `linear` affect only the top layer, and `model.v[0]` remains the surface velocity in `km/s`; `sqrt` uses `a` in `m/s/sqrt(m)` as `v_m/s = v0_m/s + a * sqrt(z_m)`, while `linear` uses `a` as slope in `m/s/m` as `v_m/s = v0_m/s + a * z_m`. Set `top_layer_gradient: all_linear` to invert one linear slope per finite layer; then `model.v[i]` is the velocity at the top of layer `i`, `model.a[i]` is its slope in `m/s/m`, and `assumptions.minimum_velocity_jump_percent` can require the velocity just below each discontinuity to be at least that percent higher than the velocity just above it. The older `top_layer_sqrt_gradient: true` flag still works as an alias for `top_layer_gradient: sqrt`.
- For fixed-layer PP/SS travel-time inversions, you can constrain each discontinuity with `assumptions.velocity_transition_directions`, for example `[inc, inc, dec]`. When the sqrt top layer is enabled, the first transition compares the second-layer velocity against the top layer velocity at its bottom.
- `assumptions` can be set globally, on an event, or inside an individual experiment. Experiment-level values override only the keys you set, so you can change `enforce_increasing_velocity`, `top_layer_sqrt_gradient`, or `velocity_transition_directions` for one run without repeating the full defaults block.
