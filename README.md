# CID-Fusion

Official implementation of **CID-Fusion**, an infrared and visible image fusion method based on collaborative interaction diffusion.

CID-Fusion formulates infrared and visible image fusion as a conditional generation problem. The implementation uses a three-branch noise predictor for the visible, infrared, and fusion states, with shared-noise forward diffusion, an unsupervised fusion objective, and deterministic reverse sampling by default.

## Contents

- [Requirements](#requirements)
- [Repository layout](#repository-layout)
- [Data preparation](#data-preparation)
- [Configuration](#configuration)
- [Training](#training)
- [Inference](#inference)
- [Outputs](#outputs)
- [Notes](#notes)

## Requirements

- Python 3.9 or later
- PyTorch (install a build compatible with your CUDA version if using a GPU)
- NumPy
- Pillow
- OpenCV (`opencv-python`)
- tqdm
- TensorBoard (optional; training continues without it)

Install the non-PyTorch dependencies with:

```bash
pip install numpy pillow opencv-python tqdm tensorboard
```

Install PyTorch by following the instructions for your operating system and CUDA version at [pytorch.org](https://pytorch.org/get-started/locally/).

## Repository layout

Arrange the source files as follows. The filenames below omit any numbering that may have been added when files were uploaded or downloaded.

```text
CID-Fusion/
├── dataset.py
├── train.py
├── inference.py
├── utils.py
├── Diffusion.py
├── config.json
├── Condition_Noise_Predictor/
│   ├── __init__.py
│   └── UNet.py
├── Dataset/
│   └── your_dataset/
│       ├── VIS/
│       └── IR/
├── weight/                 # created during training
├── logs/                   # created during training
└── generate_imgs/          # created during inference
```

`Condition_Noise_Predictor/UNet.py` must be importable as `Condition_Noise_Predictor.UNet`, and the package's `__init__.py` must provide the `time_embedding`, `Downsample`, and `Upsample` components imported by `UNet.py`.

## Data preparation

The dataset loader expects each dataset directory to contain two folders:

```text
Dataset/your_dataset/
├── VIS/
│   ├── 1.png
│   └── ...
└── IR/
    ├── 1.png
    └── ...
```

Place corresponding visible and infrared images at the same relative position in the two folders. The loader sorts filenames naturally and pairs images by their sorted index, so both folders must contain the same number of correctly aligned image pairs. Images are loaded as RGB; grayscale infrared images are converted to three channels.

The repository does not require a particular directory name for a dataset. Use the dataset's permitted download/source and update the paths in `config.json` accordingly.

## Configuration

Before running, edit `config.json`:

- `dataset.train.path`: path to the training dataset root containing `VIS/` and `IR/`.
- `dataset.valid.path`: path to the inference/test dataset root containing `VIS/` and `IR/`.
- `dataset.valid.dataset_name`: name used in the generated-output directory.
- `hyperParameter.epochs`, `dataset.train.batch_size`, and `dataset.train.imgSize`: training settings.
- `diffusion_model.T`: diffusion steps (the provided configuration uses 100).
- `diffusion_model.share_vis_ir_noise`: whether visible and infrared inputs share forward-process noise.
- `diffusion_model.add_noise`: whether stochastic noise is added during reverse sampling. Set to `false` for deterministic sampling.

The uploaded example configuration contains machine-specific Windows paths. Replace them with paths valid on your machine, for example:

```json
"path": "./Dataset/MSRS/train"
```

Make sure the selected `VIS/` and `IR/` directories contain matching image pairs. The training and validation paths can point to separate splits.

## Training

From the repository root, run:

```bash
python train.py
```

Training reads `config.json` from the current directory. Model checkpoints are saved under `weight/<run_timestamp>/`, and text/TensorBoard logs are saved under `logs/<run_timestamp>/`. The checkpoint filename follows the form `epoch_<n>.pt`.

To resume or initialize from a checkpoint, set `Condition_Noise_Predictor.use_preTrain_model` to `true` and provide the checkpoint path in `Condition_Noise_Predictor.preTrain_Model_path`.

## Inference

1. Set `dataset.valid.path` and `dataset.valid.dataset_name` in `config.json`.
2. In `inference.py`, set `model_path` in the `__main__` block to the checkpoint to evaluate.
3. Run from the repository root:

```bash
python inference.py
```

Inference uses the diffusion and model settings from `config.json`. The provided configuration uses 512 × 512 test images and deterministic reverse sampling (`add_noise: false`). Adjust the resize settings if your input resolution differs.

## Outputs

Inference creates the following directory structure:

```text
generate_imgs/<dataset_name>/<run_timestamp>/<checkpoint_name>/
├── fused/
├── ir/
└── vis/
```

The fused results are written to the `fused/` subdirectory. The visible and infrared branches are also saved for inspection. Training checkpoints and logs are stored in `weight/` and `logs/`, respectively.

## Notes

- Run commands from the repository root so relative paths such as `config.json`, `weight/`, and `generate_imgs/` resolve correctly.
- The example `config.json` is a starting point; adapt dataset paths, image sizes, batch size, and checkpoint paths to your environment.
- Set random seeds and record the software/hardware versions if you need strict experiment reproducibility; the current scripts do not configure all random-number generators or save a complete environment lockfile.

## Citation

If you use this code in academic work, please cite the corresponding CID-Fusion paper. 
