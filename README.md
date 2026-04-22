# MarPPSS Workflow

This is a quick-start guide for the current config-driven workflow.

## 1. Define paths

Set these in your shell before running anything:

```bash
export MARPPSS_DATA_DIR="/Users/evanzhang/zzq@umd.edu - Google Drive/My Drive/Research/MarPPSS"
export MARPPSS_CODE_DIR="/Users/evanzhang/Documents/Research/MarPPSS"
```

`MARPPSS_DATA_DIR` is the root directory that contains the `data/` and `run/` folders.

`MARPPSS_CODE_DIR` is the path to this repository.

## 2. Choose the config file

The example workflow config is:

```bash
$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml
```

## 3. List available experiments

```bash
python -m marppss list "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

This prints both configured events and configured experiments.

## 4. Prep the event once

This reads the SAC files, applies filtering/windowing, builds the reference wavelet, and writes processed files into `data/`.

Example:

```bash
python -m marppss prep "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --event S0976asdr_1883
```

You only need to do this once for a given event/preprocessing setup.

## 5. Run one experiment

Example:

```bash
python -m marppss run "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --experiment s0976asdr_1883_gv_1to40_20
```

Another example:

```bash
python -m marppss run "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" --experiment s0976asdr_1883_pp_tt_gv_20_30
```

## 6. Run multiple experiments in batch

Dry run first:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_ \
  --dry-run
```

Run all `GV only` experiments:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_
```

Run all `PP TT + GV` experiments:

```bash
python "$MARPPSS_CODE_DIR/scripts/run_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_pp_tt_gv_
```

## 7. Plot one run

Example for a `GV only` run:

```bash
python -m marppss plot \
  "$MARPPSS_DATA_DIR/run/S0976asdr_1883_src_3.5_s_SS/run_gv_1to40_20" \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

Example for a `PP TT + GV` run:

```bash
python -m marppss plot \
  "$MARPPSS_DATA_DIR/run/S0976asdr_1883_src_3.5_s_PP/run_pp_tt_gv_20_30" \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
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

Plot all `GV only` runs:

```bash
python "$MARPPSS_CODE_DIR/scripts/plot_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_gv_ \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
```

Plot all `PP TT + GV` runs:

```bash
python "$MARPPSS_CODE_DIR/scripts/plot_batch.py" \
  "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml" \
  --prefix s0976asdr_1883_pp_tt_gv_ \
  --reference-config "$MARPPSS_CODE_DIR/configs/mars_workflow.example.yaml"
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
