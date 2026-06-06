# Meteorite Binary Image Classification

This repository contains the code release for the SUSTech Data Science Practice meteorite binary image classification project.

## Task

The task is binary image classification:

- label=1: meteorite
- label=0: non-meteorite

The evaluation metric is F1 score.

Final Kaggle public F1 score:

0.84571

Final submitted CSV:

final_submission/01_ensemble_5fold_best_f1_F1_0.994350_pos89_FP1_FN0.csv

## Repository contents

- train_cv.py: 5-fold training script
- inference.py: 5-fold checkpoint inference and submission generation
- dataset.py: dataset and image loading utilities
- models.py: model definition utilities
- utils.py: shared helper utilities
- configs/final_config.yaml: final configuration used for the released model
- scripts/run_round2curated_056119131184pos_114neg_train5fold.sh: optional final 5-fold training command
- docs/DATA.md: data and checkpoint notes

## Environment

Recommended environment:

- Python 3.10 or compatible
- PyTorch
- timm
- pandas
- numpy
- scikit-learn
- opencv-python
- PyYAML
- tqdm

Install dependencies with:

    pip install -r requirements.txt

## Model checkpoint package

Trained checkpoints are not included in this GitHub repository because each checkpoint is large.

Download the model package from:

Baidu Netdisk link: https://pan.baidu.com/s/1ahdqqZy_oSnC91Rde_R0Rw?pwd=quc3

Extraction code: quc3

The model package contains the released checkpoints. For reproducing the final submitted CSV, use:

    checkpoints_best_f1/fold0_best_f1.pth
    checkpoints_best_f1/fold1_best_f1.pth
    checkpoints_best_f1/fold2_best_f1.pth
    checkpoints_best_f1/fold3_best_f1.pth
    checkpoints_best_f1/fold4_best_f1.pth

The inference script accepts checkpoint names matching fold*_best.pth, fold*_best_f1.pth, or fold*_best_loss.pth.

## Reproduce the final submission CSV

This is the main reproduction path for the final Kaggle submission.

Step 1. Clone this repository.

Step 2. Put the official test files under the repository root:

    sample_submission.csv
    test_images/

Step 3. Download and extract the model package from Baidu Netdisk.

Recommended directory layout:

    meteorite-binary-classification/
        inference.py
        configs/final_config.yaml
        sample_submission.csv
        test_images/
        final_submission/

    model_release_for_cloud_0.84571_20260606_232402/
        checkpoints_best_f1/
            fold0_best_f1.pth
            fold1_best_f1.pth
            fold2_best_f1.pth
            fold3_best_f1.pth
            fold4_best_f1.pth

Step 4. Run inference with the released checkpoint ensemble:

    python inference.py --config configs/final_config.yaml --checkpoint-dir ../model_release_for_cloud_0.84571_20260606_232402/checkpoints_best_f1 --output reproduced_submission.csv --threshold 0.125 --tta

The final submission was generated with the 5 best_f1 checkpoints, threshold=0.125, and TTA enabled.

The generated CSV can be compared with the released final CSV by running:

    python3 - <<'PY'
    import pandas as pd
    a = pd.read_csv('reproduced_submission.csv')
    b = pd.read_csv('final_submission/01_ensemble_5fold_best_f1_F1_0.994350_pos89_FP1_FN0.csv')
    print('rows:', len(a))
    print('same shape:', a.shape == b.shape)
    print('same labels:', a.equals(b))
    print('positives:', int(a['label'].sum()))
    PY

If the same test images, sample submission file, configuration, released checkpoints, threshold=0.125, and TTA are used, the reproduced CSV should match the released final submission CSV.

The released final submission CSV was submitted to Kaggle and obtained public F1 = 0.84571.

## Optional full training

Full training is not required for reproducing the final submitted CSV from the released checkpoints.

To rerun training, the following additional files/directories must be restored according to configs/final_config.yaml:

- train_images/
- train_labels.csv
- curated external training CSV
- local ConvNeXt pretrained safetensors checkpoint

The external training CSV should contain at least these columns:

- id
- path
- label
- sample_weight

The path column must point to image files that are accessible in the current environment.

After restoring all required training assets, training can be launched with:

    bash scripts/run_round2curated_056119131184pos_114neg_train5fold.sh

## Release notes

This repository is organized for code review and final submission inference reproduction.

The GitHub release intentionally excludes raw images, external image data, curated external image data, trained checkpoints, large outputs/logs, API keys, and private credentials.

