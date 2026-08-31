# CoNNGaFit
<b>Co</b>nvoltuional <b>N</b>eural <b>N</b>etwork for <b>Ga</b>laxy <b>Fit</b>ting. Train and use convolutional neural networks to fit various galactic parameters to synthetic HI images. Training images and annotations for the FIRE simulations can be generated using VeryObservableFIRE

## Requirements

```
pip install torch torchvision numpy pandas h5py matplotlib scipy astropy
```

## Data format

Each of `CoNNGaFit_Datasets.py`, `CoNNGaFit_Datasets_600x600.py`, `CoNNGaFit_Datasets_60x60.py`, `CoNNGaFit_Datasets_1d.py` defines a `CoNNGaFitImageDataset` that reads a CSV with no header:

- Column 0: path (relative to the dataset's `root_dir`, normally `.`) to the input datacube - an `.hdf5` file containing a `spectra` dataset (or `moments`, if `use_moment_maps=True`), a `.fits` file, or a plain image.
- Columns 1..N: the flattened target label map. N must match `Nlabels` at the top of the dataset file being used (`40*40` for the standard/HiRes-test networks, `600*600` for the 600x600 network, `60*60` for the 60x60 variant).

Training/validation/test CSVs are normally produced by a `WriteDatasetsToCsv`-style script (see below) - training scripts expect these to already exist.

## Directory layout

The standard training scripts (all but the HiRes-test-2 script) expect, and can be pointed elsewhere with `--data-dir`/`--network-dir`:

```
<data-dir>/                      (default: CoNNGaFitData)
  annotation_datasets/
    training_annotations_*.csv
    validation_annotations_*.csv
    test_annotations_*.csv
  outputs/
    images/                      (per-epoch comparison plots/hdf5 snapshots)
    diagnostics/                 (loss curves, correlation plots)
<network-dir>/                   (default: TrainedNetworks)
  <model-name>.pt                (trained weights)
  <model-name>.hdf5              (normalization stats + hyperparameters)
```

`<data-dir>/outputs/images` and `<data-dir>/outputs/diagnostics` are not created automatically - create them before running a training script or `plt.savefig` will fail.

## Training

Four training scripts are available, each pairing a training loop with one network implementation:

| Script | Network used | Target |
|---|---|---|
| `CoNNGaFit_TrainModel_MassFlux_UNet.py` | `CoNNGaFit_NeuralNetwork_Unet3d.py` | Radial mass flux (40x40) |
| `CoNNGaFit_TrainModel_RC_UNet.py` | `CoNNGaFit_NeuralNetwork_Unet3d.py` | Rotational velocity (40x40) |
| `CoNNGaFit_TrainModel_MassFlux_UNet_HiResTest.py` | `AlternativeNetworks/CoNNGaFit_NeuralNetwork_Unet3d_HiResTest.py` | Radial mass flux (40x40, lighter stem) |
| `CoNNGaFit_TrainModel_MassFlux_UNet_HiResTest2.py` | `CoNNGaFit_NeuralNetwork_Unet3d_600x600.py` | Radial mass flux (600x600) |

All four are run directly with Python and take command-line options via `argparse` - run any of them with `--help` to see the full list. Example:

```
python CoNNGaFit_TrainModel_MassFlux_UNet.py --data-dir CoNNGaFitData --sample-suffix All_Inclinations_finalSnapNoM12m
```

### Common options (first three scripts)

| Flag | Default | Purpose |
|---|---|---|
| `--sample-suffix` | script-specific | Suffix used to build default CSV/output/model filenames, unless overridden below |
| `--data-dir` | `CoNNGaFitData` | Root for `annotation_datasets/` (input) and `outputs/` (diagnostics/images) |
| `--network-dir` | `TrainedNetworks` | Where the trained `.pt`/`.hdf5` checkpoint is written |
| `--training-csv` / `--validation-csv` / `--testing-csv` | built from `--sample-suffix` | Explicit annotation CSV filenames, read from `<data-dir>/annotation_datasets/` |
| `--output-name` | built from `--sample-suffix` | Base name for diagnostic images/plots |
| `--model-name` | built from `--sample-suffix` | Base filename for the saved checkpoint |

`CoNNGaFit_TrainModel_MassFlux_UNet_HiResTest2.py` uses a different (flatter) layout and default `--sample-suffix HiResTest`, since it doesn't use the `CoNNGaFitData` convention:

| Flag | Default |
|---|---|
| `--data-dir` | `./training_datasets` (also where `--training-csv`/etc. are read from) |
| `--output-dir` | `./outputs` |
| `--network-dir` | `./networks` |
| `--training-csv` | `training_annotations_HiRes_MassFlux.csv` |
| `--validation-csv` / `--testing-csv` | same file as `--training-csv`, unless given explicitly |

### Hyperparameters

Not exposed on the command line - edit the `HYPERPARAMETERS` block near the top of the script you're running:

- `learning_rate`, `weight_decay`, `epochs` - standard training controls (Adam optimizer).
- `nFilt0`, `k0`, `k1`, `nFC` - network size: initial filter count, initial/block kernel sizes, and the size of the optional FC bottleneck (set `nFC = 0` to remove it).
- `dropout_rate` (default `0.3`) - dropout applied just before the output layer.
- `block_dropout_rate` (default `0.0`, off) - channel-wise dropout inside every residual/deconv block; see the in-file comment before enabling.
- `batchSizeDefault` - `None` trains on the full dataset as a single batch each epoch (used by three of the four scripts); set to an integer for mini-batch training (used by the HiRes-test-2 script, `batchSizeDefault = 5`).

### Output

Each run writes `<model-name>.pt` (weights) and `<model-name>.hdf5` (image normalization stats + the hyperparameters above, needed to reconstruct the model later) to `--network-dir`, plus loss/accuracy curves to `<data-dir>/outputs/diagnostics/` (or `--output-dir` for the HiRes-test-2 script) and per-epoch comparison images/hdf5 snapshots to `<data-dir>/outputs/images/` on the final training epoch.

## Building training datasets

`WriteDatasetsToCsv_HiResTests.py` scans a directory of FIRE-2 simulation outputs and writes a training-annotations CSV in the format above:

```
python WriteDatasetsToCsv_HiResTests.py --sim-dir /path/to/fire-2 --output-dir /path/to/output
```

Run with `--help` for the full flag list (`--output-name`, `--use-denoised-spectra`, `--use-time-averaged-annotations`); the galaxy/snapshot/inclination/position-angle lists to scan are set as in-file variables near the bottom of the script rather than CLI flags.

To convert an observed FITS datacube into a CoNNGaFit-compatible `.hdf5` input, use `FITS_to_CoNNGaFit(imageDir, output, targetNpix, targetNspec, distance=-1, saveImages=False)` from `FITS_to_CoNNGaFit.py` - see `FITS_to_CoNNGaFit_Template.py` for example usage (edited directly, no CLI).

## Inference

`CoNNGaFit_UseModel.RunInferences(networkType, imageOutput_prefix, imageList, modelPath=None, params=None, saveLatentImages=False)` loads a trained checkpoint and runs it over a CSV list of input datacubes (same one-path-per-line format as training, labels ignored if present). Currently only `networkType='unet18'` is implemented. `modelPath` overrides the built-in default checkpoint path; `params=[nFilt0, k0, k1, nFC]` overrides the default architecture hyperparameters.

Two ready-to-run scripts wrap this with `argparse`:

```
python InferenceTemplate.py --image-list path/to/list.csv --output-prefix path/to/output_ --model-path path/to/checkpoint
python InferenceTest_ValidationSet.py --sample-suffix finalSnapNoM12m --model-path path/to/checkpoint
```

Run either with `--help` for the full flag list. `InferenceTest_ValidationSet.py` runs inference on a sample's validation-annotations CSV specifically, to spot-check predictions against known labels.
