# Shifting the retinal foundation models paradigm from slices to volumes for optical coherence tomography

[![Python 3.11](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)

For methodological details and citation information, please refer to our [PAPER](https://www.nature.com/articles/s41746-026-02496-7):

![concept](https://github.com/aim-lab/oct-fm-slices-to-volumes/blob/master/Summarys.png?raw=true?as=webp)
## Overview

This repository contains the code required to reproduce the experiments presented in our paper.

The objective of this work is to benchmark retinal foundation models under two distinct paradigms for Optical Coherence Tomography (OCT):

- **Slice-based modeling** (single B-scan input)
- **Volume-based modeling** (multi-slice / video-style input)

We systematically evaluate whether leveraging **video foundation models** for volumetric OCT provides performance gains compared to traditional single-slice approaches.

Our study benchmarks:

- 4 foundation models  
- 5 downstream tasks  
- Multiple OCT datasets with heterogeneous acquisition protocols  

The central finding is that **volumetric (video-style) modeling improves downstream performance by enabling inter-slice interaction learning**, compared to single-slice training.

## Repository Features

This repository provides:

- Model loading utilities  
- Dataset loaders for multiple OCT cohorts  
- Fine-tuning scripts supporting slice and volumetric aggregation  
- Reproducible experiment configurations  

---


Supported training aggregation modes:

- `scan`
- `volume`

Supported model entrypoints:

- `local_retfound`
- `retfound`
- `dinov2`
- `visionfm`
- `vjepa`

## Quick Start

### 1) Environment Setup

The framework is based on:

- **Python 3.11**
- **PyTorch 2.8.0** (tested)

Create the environment:

```bash
conda create -n oct_volumes python=3.11 -y
conda activate oct_volumes
pip install -r requirements.txt
```

### 2) Training example
Example: fine-tuning DINOv2 on CirrusOCT using single-slice aggregation.
```bash

python src/run/train/finetune.py \
  --model dinov2 \
  --finetune_vit facebook/dinov2-large \
  --aggregate scan \
  --num_frames 1 \
  --dataset-name cirrusoct \
  --data_path data/CirrusOCT/

```
- `--model`  
  Selects the model wrapper / entrypoint.

- `--finetune_vit`  
  Specifies the pretrained weights to load.

- `--aggregate`  
  Defines the training paradigm:
  - `scan` → single B-scan (2D input)
  - `volume` → multi-slice volumetric input (video-style)

- `--num_frames`  
  Must be consistent with the aggregation mode:
  - `1` for slice-based training  
  - `>1` for volumetric training  

- `--dataset-name`  
  Selects dataset-specific parsing logic.

---

---

## Supported Model Entrypoints
  
- `retfound`  
- `dinov2`  
- `visionfm`  
- `vjepa`  

Each entrypoint supports loading pretrained weights and adapting them for downstream OCT tasks.

---

## Data

⚠️ This repository **does not include any medical data**.

You must independently obtain access to the relevant OCT datasets.

---


### Dataset Folder Structure (high-level)

Actual parsing logic is implemented in `src/datasets/*.py`;
Verify the corresponding loader before organizing your data.

### Example High-Level Structures

- **Gamma**
  - `Train/Validation/Test`
  - `multi-modality_images/...`
  - Associated label spreadsheets

- **Neh_ut**
  - Condition folders (`NORMAL`, `DRUSEN`, `CNV`)
  - Scan subfolders per subject

- **A2A**
  - `Control/AMD`
  - `.mat` volumetric files

- **CirrusOCT**
  - Dataset-specific hierarchy handled in:
    ```
    src/datasets/CirrusOCT.py
    ```

---

## Outputs

- Checkpoints: `trained_models/<model>/<task>/...`
- Logs/metrics/predictions: `logs/<model>/<task>/...`


## License

This project is licensed under the MIT License. See [LICENSE](LICENSE).


## Data and Privacy Notice

This repository does **not** redistribute any patient or medical data.

Users are responsible for:

- Ensuring lawful access to datasets  
- Complying with data-sharing agreements  
- Performing proper de-identification where required  

---

## Citations

If you use this code, please cite our paper:

```bibtex
@article{article,
author = {Judkiewicz, Raphael and Berkowitz, Eran and Meisel, Meishar and Michaeli, Tomer and Behar, Joachim},
year = {2026},
pages = {},
title = {Shifting the retinal foundation models paradigm from slices to volumes for optical coherence tomography},
journal = {npj Digital Medicine},
publisher={Nature Publishing Group},
}
```
