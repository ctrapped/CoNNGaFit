# Training walkthrough

Short guide to training a CoNNGaFit network on your personal computer or GPU cluster.

## 1. Environment setup

### Personal Computer

Install the requisite python packages below. Make sure to use a CUDA-enabled version of `pytorch` build if training on GPU. 


```
pip install torch torchvision numpy pandas h5py matplotlib scipy astropy
```


### GPU Cluster


For supported GPU clusters (currently only Quartz at IU), follow the instructions in the appropriately named subfolder (i.e. Quartz). You will also find example job scripts for training on said clusters.



## 2. Data layout

The standard training scripts expect, relative to `--data-dir` (default
`CoNNGaFitData`):

```
<data-dir>/
  annotation_datasets/
    training_annotations_MassFlux_<suffix>.csv
    validation_annotations_MassFlux_<suffix>.csv
    test_annotations_MassFlux_<suffix>.csv
  outputs/images/          <- create manually
  outputs/diagnostics/     <- create manually
<network-dir>/             (default TrainedNetworks)
```

`outputs/images` and `outputs/diagnostics` are **not** created automatically —
`mkdir` them first or `plt.savefig` will fail.

Each annotation CSV has no header: column 0 is the path to an input datacube
(`.hdf5` with a `spectra` dataset, `.fits`, or a plain image), resolved relative
to the dataset `root_dir` (normally `.`, i.e. the launch directory); columns
1..N are the flattened target label map (N must match `Nlabels` in the dataset
file — `40*40` for the standard networks). CSVs are produced by a
`WriteDatasetsToCsv`-style script; the training scripts expect them to exist.

## 3. Pick a script

| Script | Target |
|---|---|
| `TrainModel_MassFlux_UNet.py` | Radial mass flux (40×40) |
| `TrainModel_RC_UNet.py` | Rotational velocity (40×40) |
| `TrainModel_MassFlux_UNet_HiResTest.py` | Radial mass flux, lighter stem |
| `TrainModel_MassFlux_UNet_HiResTest2.py` | Radial mass flux (600×600), flat dir layout |

## 4. Set hyperparameters

Not CLI flags — edit the `HYPERPARAMETERS` block near the top of the script:
`learning_rate`, `weight_decay`, `epochs` (default 4000; drives wall time),
`nFilt0`/`k0`/`k1`/`nFC` (network size; `nFC=0` removes the FC bottleneck),
`dropout_rate`, `block_dropout_rate`, `batchSizeDefault` (`None` = full dataset
as one batch each epoch, used by the first three scripts).

## 5. Run

```bash
python TrainModel_MassFlux_UNet.py \
    --data-dir CoNNGaFitData \
    --network-dir TrainedNetworks \
    --sample-suffix All_Inclinations_finalSnapNoM12m
```

`--sample-suffix` builds the default CSV / output / model names; override any
individually with `--training-csv` / `--validation-csv` / `--testing-csv` /
`--output-name` / `--model-name`. Run with `--help` for the full list.

The `HiResTest2` script uses a different layout: `--data-dir ./training_datasets
--output-dir ./outputs --network-dir ./networks` plus explicit `--*-csv`.

## 6. Outputs

- `<network-dir>/<model-name>.pt` — trained weights.
- `<network-dir>/<model-name>.hdf5` — normalization stats + hyperparameters,
  needed to rebuild the model for inference.
- `<data-dir>/outputs/diagnostics/` — loss / accuracy / correlation plots.
- `<data-dir>/outputs/images/` — per-epoch comparison images / hdf5 snapshots
  (written on the final epoch).

See the top-level `README.md` "Inference" section for using the checkpoint.
