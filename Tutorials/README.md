# Training walkthrough

Short guide to training a CoNNGaFit network on your personal computer or a GPU cluster, from VOF outputs to a trained model.

## 1. Environment setup

### Personal computer

From the repository root:

```bash
pip install -e .
```

This installs CoNNGaFit's dependencies and lets Python import its modules from this folder. To train on a GPU, install a CUDA-enabled build of PyTorch first (see [pytorch.org](https://pytorch.org/get-started/locally/)).

### GPU cluster

For supported GPU clusters (currently only Quartz at IU), follow the instructions in the matching subfolder, e.g. [`Quartz/`](Quartz/README.md), which also has an example SLURM job script.

## 2. Generate training data with VOF

CoNNGaFit trains on the optically thin outputs of [VeryObservableFIRE](https://github.com/ctrapped/VeryObservableFIRE) (VOF): one `.hdf5` file per galaxy, snapshot and viewing angle, holding the spectral datacube (`spectra`) and the true gas properties at every pixel. VOF writes them as:

```
<VOF output directory>/
  i45/
    m12i_i45_pa0_600_image_fullSpectra.hdf5
    m12i_i45_pa90_600_image_fullSpectra.hdf5
    ...
  i0/
    ...
```

Run VOF for each galaxy you want to train, validate or test on, using the inclinations you want to train on. To train on radial curves measured face on (`radial_projection = "face_on"`, below), also include inclination 0: each image's curve is read from its snapshot's `i0` image.

## 3. Write a datasets file

Copy `configs/datasets.toml` and edit it. It sets what to learn and which VOF output directories to learn it from:

```toml
target = "radial_momentum"
augment = "major_axis"
inclinations = [45]
radial_bins = 0
radial_projection = "face_on"
base_dir = "/path/to/VOF_outputs"   # optional: relative paths below are relative to this

[training]
"m12i" = [500, 600]   # VOF output directory = a snapshot, or an inclusive [min, max] range
"m12f" = 600

[validation]
"m12c" = 600

[testing]
"m12m" = 600
```

- `target`: `mass`, `radial_velocity`, `cylindrical_radial_velocity`, `rotational_velocity`, `radial_momentum`, `cylindrical_radial_momentum` or `rotational_momentum`.
- `augment`: rotations/flips added to the training set: `"major_axis"` (4 per image, keeping the major axis along x), `"all"` (8 per image) or `"none"`.
- `inclinations`: `"all"`, or a list in degrees, used for every set.
- `radial_bins`: `0` trains on maps (one value per pixel); `N > 0` trains on `N`-bin radial curves.
- `suffix` (optional): VOF's `outputSuffix`, if that VOF run used one.

Every key is checked when training starts, and a missing directory, snapshot or inclination stops the run with an error naming it.

## 4. Choose hyperparameters

`configs/hyperparams.toml` (tuned for radial momentum) and `configs/hyperparams_rc.toml` (rotational velocity) are the starting points; the comments in each file explain every key. The ones you'll most often change:

- `epochs`: training length, and the main driver of run time.
- `output_head`: `"conv_halfres"` or `"conv"` for large maps (128×128 and up), `"fc"` for small maps or radial curves.
- `n_filt0`, `k0`, `k1`: network size.
- `label_scaling`: `"symmetric"` for signed targets like radial momentum, `"one-sided"` for targets like rotational velocity.

## 5. Run

```bash
python train_model_unet.py \
    --hyperparams configs/hyperparams.toml \
    --datasets my_datasets.toml \
    --run-name m12_momentum
```

Run from the repository root. The defaults for `--hyperparams` and `--datasets` are the files in `configs/`. `--run-name` names the checkpoint and every output (default: the target). Batch sizes default to about a tenth of each set; override them with `--batch-size` and `--test-batch-size`. Run with `--help` for every option.

Progress is logged with timestamps (`-v` for debug detail, `-q` for warnings only). Log messages go to stderr, so on a cluster make sure your job script keeps it, e.g. with `#SBATCH --error` as in the Quartz example.

## 6. Outputs

- `NetworkWeights/<run-name>.pt`: trained weights.
- `NetworkWeights/<run-name>.hdf5`: everything needed to rebuild and use the network, including the image normalization (`image_stats`), label scaling (`label_stats`), input shape, output head and the full text of both parameter files.
- `DiagnosticOutputs/diagnostics/`: loss and accuracy curves, prediction/truth correlation plots, and `<run-name>_LossesAndAccuracies.hdf5`.
- `DiagnosticOutputs/images/`: final-epoch prediction vs. truth for every sample (`.hdf5`, plus a comparison `.png` for validation and test samples), in physical units.

Change the two folders with `--network-dir` and `--output-dir`; both are created if missing.
