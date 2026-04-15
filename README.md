# Multimodal Medical Predictions

This repository contains experiments for landmark-based medical image prediction with optional metadata fusion. The main workflow trains models to predict landmark heatmaps from images, then evaluates the predicted landmarks with task-specific comparison metrics.

The codebase includes:

- image-only and multimodal models (dataloader set up but not current model)
- U-Net, U-Net++, HRNET style architectures
- experiment-driven configuration via YAML files
- training, testing, visualisation, and temperature-scaling utilities

Some landmark-processing ideas in this project were adapted from James McCouat's work: <https://github.com/jfm15/>

## What The Models Predict

The dataloader provides:

- an input image
- landmark targets represented as heatmaps
- optional metadata features

The model output is a stack of landmark heatmaps. Downstream evaluation extracts the hottest point from each predicted heatmap and compares it against the reference annotations.

## Repository Layout

```text
.
├── experiments/              # YAML experiment configs
├── utils/
│   ├── preprocessing/        # data loading and augmentation
│   ├── main/                 # model init, training, validation, testing, metrics
│   ├── support/              # config + logging helpers
│   └── visualisations/       # plotting/comparison utilities
├── data/                     # example partition files
├── run_train_*.sh            # cluster job scripts for training
├── run_test*.sh              # cluster job scripts for testing
└── environment.yml           # Conda environment
```

## Environment Setup

Create the Conda environment:

```bash
conda env create -f environment.yml
conda activate mm_env
```

This project expects PyTorch, TensorFlow, OpenCV, `yacs`, and related scientific Python packages to be available through that environment.

## Running Experiments

# Experiment Configuration

Important: `--cfg` should be the experiment name without the `.yaml` extension. The script loads configs from `experiments/<name>.yaml`.

Experiment files live in [`experiments/`](/home/allent/Desktop/repos/MultimodalMedicalPredictions/experiments). A typical config defines:

- `INPUT_PATHS`: dataset paths, metadata CSV, partition JSON, and label/image folders
- `MODEL`: architecture name and fusion settings
- `DATASET`: image size, number of landmarks, augmentation, and annotation handling
- `TRAIN`: batch size, learning rate, epochs, loss, and regularisation
- `TEST`: checkpoint path, metrics, output options, and dataloader settings
- `OUTPUT_PATH`: where logs, plots, and trained weights are written

Useful defaults are defined in [`utils/support/default_config.py`](/home/allent/Desktop/repos/MultimodalMedicalPredictions/utils/support/default_config.py).

### Training
Example:

```bash
python utils/run_training.py --cfg ddh_arc_newsplits_0.01466
```
See config for input details.
### Testing

```bash
python utils/run_test.py --cfg ddh_arc_newsplits_0.01466
```

## Outputs

Training produces:

- log files
- loss plots
- saved model weights under `OUTPUT_PATH/model:<n>/`

Testing can optionally produce:

- predicted-vs-true landmark visualisations
- saved heatmaps
- text outputs
- metric summaries

## HRNet Setup

Some experiments depend on HigherHRNet, which is not vendored into this repository. 

Clone it alongside this repo:

```bash
cd ..
git clone git@github.com:HRNet/HigherHRNet-Human-Pose-Estimation.git
```

This project also expects a manual edit in `lib/models/pose_higher_hrnet.py` inside the cloned HRNet repo:

```python
def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHigherResolutionNet(cfg, **kwargs)

    if is_train and cfg.MODEL.INIT_WEIGHTS:
        model.init_weights(cfg.MODEL.PRETRAINED, verbose=1)

    return model
```

This is needed because the current integration assumes `verbose=1` is passed directly rather than configured elsewhere.

NOTE: image size input must be 512 by 512 

## Notes

- Most paths in the experiment YAML files are environment-specific and will need to be updated locally.
- Several shell scripts in the repo are SLURM job submission wrappers that show the intended cluster usage.
- [`experiments/README.md`](/home/allent/Desktop/repos/MultimodalMedicalPredictions/experiments/README.md) contains brief notes on individual experiment variants.
