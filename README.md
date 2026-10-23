# CoNNGaFit
[![tests](https://github.com/ctrapped/CoNNGaFit/actions/workflows/tests.yml/badge.svg)](https://github.com/ctrapped/CoNNGaFit/actions/workflows/tests.yml)

<b>Co</b>nvoltuional <b>N</b>eural <b>N</b>etwork for <b>Ga</b>laxy <b>Fit</b>ting. Train and use convolutional neural networks to fit various galactic parameters to synthetic HI images. Training images and annotations for the FIRE simulations can be generated using VeryObservableFIRE

## Requirements

```
pip install torch torchvision numpy pandas h5py matplotlib scipy astropy
```

## Data format

Each of `conngafit_datasets.py`, `Datasets_600x600.py`, `Datasets_60x60.py`, `Datasets_1d.py` defines a `CoNNGaFitImageDataset` that reads a CSV with no header:

- Column 0: path (relative to the dataset's `root_dir`, normally `.`) to the input datacube - an `.hdf5` file containing a `spectra` dataset (or `moments`, if `use_moment_maps=True`), a `.fits` file, or a plain image.
- Columns 1..N: the flattened target label map. N must match `Nlabels` at the top of the dataset file being used (`40*40` for the standard/HiRes-test networks, `600*600` for the 600x600 network, `60*60` for the 60x60 variant).

Training/validation/test CSVs are normally produced by a `WriteDatasetsToCsv`-style script (see below) - training scripts expect these to already exist.

## Directory layout

All four training scripts share the same three-directory convention and the same defaults, so with data already
in place, **the only argument you normally need to pass is `--sample-suffix`**:

```
TrainingData/                    (input - override with --data-dir)
  annotation_datasets/
    training_annotations_*.csv
    validation_annotations_*.csv
    test_annotations_*.csv
NetworkWeights/                  (output - override with --network-dir; created automatically)
  <model-name>.pt                (trained weights)
  <model-name>.hdf5              (normalization stats + hyperparameters)
DiagnosticOutputs/                (output - override with --output-dir; created automatically)
  images/                        (per-epoch comparison plots/hdf5 snapshots)
  diagnostics/                   (loss curves, correlation plots)
```

`TrainModel_MassFlux_UNet_HiResTest2.py` is the one exception: its CSVs sit directly in `<data-dir>` with no
`annotation_datasets/` subfolder (see below). `<network-dir>` and `<output-dir>/{images,diagnostics}` are created
automatically on every run if they don't already exist; `<data-dir>` (input) is not.

## Training

Four training scripts are available, each pairing a training loop with one network implementation:

| Script | Network used | Target |
|---|---|---|
| `TrainModel_MassFlux_UNet.py` | `NetworkModels/NeuralNetwork_Unet3d.py` | Radial mass flux (40x40) |
| `TrainModel_RC_UNet.py` | `NetworkModels/NeuralNetwork_Unet3d.py` | Rotational velocity (40x40) |
| `TrainModel_MassFlux_UNet_HiResTest.py` | `NetworkModels/AlternativeNetworks/Unet3d_HiResTest.py` | Radial mass flux (40x40, lighter stem) |
| `TrainModel_MassFlux_UNet_HiResTest2.py` | `NetworkModels/NeuralNetwork_Unet3d_600x600.py` | Radial mass flux (600x600) |

All four are run directly with Python and take command-line options via `argparse` - run any of them with `--help`
to see the full list, grouped as below. Typical usage:

```
python TrainModel_MassFlux_UNet.py --sample-suffix All_Inclinations_finalSnapNoM12m
```

### `--sample-suffix`

The one option you're expected to pass routinely. It's used to build every default CSV/output/model filename
below, so as long as `TrainingData/annotation_datasets/` holds annotation CSVs named for it, nothing else needs
to be given.

### Input/output directories (override the default layout)

| Flag | Default | Purpose |
|---|---|---|
| `--data-dir` | `TrainingData` | Root annotation CSVs are read from (`<data-dir>/annotation_datasets/`) |
| `--network-dir` | `NetworkWeights` | Where the trained `.pt`/`.hdf5` checkpoint is written; created automatically |
| `--output-dir` | `DiagnosticOutputs` | Root for diagnostic images (`<output-dir>/images/`) and plots (`<output-dir>/diagnostics/`); created automatically |

`TrainModel_MassFlux_UNet_HiResTest2.py` reads CSVs directly from `<data-dir>` instead of
`<data-dir>/annotation_datasets/`, and defaults `--sample-suffix` to `HiResTest`; its directory flags otherwise
default the same as above.

### Explicit filenames (override individual files derived from `--sample-suffix`)

| Flag | Default | Purpose |
|---|---|---|
| `--training-csv` / `--validation-csv` / `--testing-csv` | built from `--sample-suffix` | Annotation CSV filenames, read from `<data-dir>` (see above) |
| `--output-name` | built from `--sample-suffix` | Base name for diagnostic images/plots |
| `--model-name` | built from `--sample-suffix` | Base filename for the saved checkpoint |

### Training options (override values computed from the dataset)

| Flag | Default | Purpose |
|---|---|---|
| `--batch-size` | ~1/10 of the training set (floored at 2) | Mini-batch size for training; `0` trains on the full dataset as a single batch per epoch |
| `--test-batch-size` | ~1/10 of the validation set (floored at 1) | Mini-batch size for validation/testing |

`TrainModel_MassFlux_UNet_HiResTest2.py` defaults `--training-csv` to the fixed filename
`training_annotations_HiRes_MassFlux.csv` rather than one built from `--sample-suffix`, and defaults
`--validation-csv`/`--testing-csv` to reusing whatever `--training-csv` resolves to.

### Hyperparameters

Not exposed on the command line (batch sizes are the exception - see the table above) - edit the
`HYPERPARAMETERS` block near the top of the script you're running:

- `learning_rate`, `weight_decay`, `epochs` - standard training controls (Adam optimizer).
- `n_filt0`, `k0`, `k1`, `n_fc` - network size: initial filter count, initial/block kernel sizes, and the size of the optional FC bottleneck (set `n_fc = 0` to remove it).
- `dropout_rate` (default `0.3`) - dropout applied just before the output layer.
- `block_dropout_rate` (default `0.0`, off) - channel-wise dropout inside every residual/deconv block; see the in-file comment before enabling.

### Output

Each run writes `<model-name>.pt` (weights) and `<model-name>.hdf5` (image normalization stats + the hyperparameters above, needed to reconstruct the model later) to `--network-dir`, plus loss/accuracy curves and correlation plots to `<output-dir>/diagnostics/` and per-epoch comparison images/hdf5 snapshots to `<output-dir>/images/` on the final training epoch.

## Building training datasets

`WriteDatasetsToCsv_HiResTests.py` scans a directory of FIRE-2 simulation outputs and writes a training-annotations CSV in the format above:

```
python WriteDatasetsToCsv_HiResTests.py --sim-dir /path/to/fire-2 --output-dir /path/to/output
```

Run with `--help` for the full flag list (`--output-name`, `--use-denoised-spectra`, `--use-time-averaged-annotations`); the galaxy/snapshot/inclination/position-angle lists to scan are set as in-file variables near the bottom of the script rather than CLI flags.

To convert an observed FITS datacube into a CoNNGaFit-compatible `.hdf5` input, use `fits_to_conngafit(image_dir, output, target_npix, target_nspec, distance=-1, save_images=False)` from `fits_to_conngafit.py` - see `fits_to_conngafit_template.py` for example usage (edited directly, no CLI).

## Inference

`use_model.run_inferences(network_type, image_output_prefix, image_list, model_path=None, params=None, save_latent_images=False)` loads a trained checkpoint and runs it over a CSV list of input datacubes (same one-path-per-line format as training, labels ignored if present). Currently only `network_type='unet18'` is implemented. `model_path` overrides the built-in default checkpoint path; `params=[n_filt0, k0, k1, n_fc]` overrides the default architecture hyperparameters.

Two ready-to-run scripts wrap this with `argparse`:

```
python inference_template.py --image-list path/to/list.csv --output-prefix path/to/output_ --model-path path/to/checkpoint
python inference_test_validation_set.py --sample-suffix finalSnapNoM12m --model-path path/to/checkpoint
```

Run either with `--help` for the full flag list. `inference_test_validation_set.py` runs inference on a sample's validation-annotations CSV specifically, to spot-check predictions against known labels.

## Citation and license

If you use CoNNGaFit in your research, please cite it using [`CITATION.cff`](CITATION.cff)
(GitHub's "Cite this repository" button). CoNNGaFit is released under the [MIT License](LICENSE).

## Contributing

`main` is protected: changes are merged through pull requests, and the tests
(`.github/workflows/tests.yml`) must pass on Python 3.11 and 3.12 first. Work on a
branch, add tests in `tests/` for new code, and run `pytest` locally before opening a PR.
Code is linted and formatted with [ruff](https://docs.astral.sh/ruff/); run `pre-commit install` once after
cloning so it runs automatically on every commit.
