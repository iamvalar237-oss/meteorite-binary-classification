# Meteorite Binary Image Classification

This repository contains the code release for the SUSTech Data Science Practice meteorite binary image classification project.

## Task

The goal is to classify test images into two classes:

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
- scripts/run_round2curated_056119131184pos_114neg_train5fold.sh: final 5-fold training command
- scripts/eval_all_round2curated_056119131184pos_114neg_vs_perfect6_1_18_49.sh: final evaluation command
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

## Model checkpoints

Trained checkpoints are not included in this GitHub repository because each checkpoint is large.

Download the model package from:

Baidu Netdisk: https://pan.baidu.com/s/1ahdqqZy_oSnC91Rde_R0Rw?pwd=quc3  Extraction code: quc3

The model package contains 10 checkpoints:

- checkpoints_best_f1/fold0_best_f1.pth to fold4_best_f1.pth
- checkpoints_best_loss/fold0_best_loss.pth to fold4_best_loss.pth

For reproducing the final submitted CSV, use the 5 checkpoints in checkpoints_best_f1/.

The inference script accepts checkpoint names matching fold*_best.pth, fold*_best_f1.pth, or fold*_best_loss.pth.

## Inference reproduction

This path is intended to reproduce the final submission using the released model weights.

Required files/directories:

- test_images/
- sample_submission.csv
- configs/final_config.yaml
- downloaded model package containing checkpoints_best_f1/

Example after extracting the model package next to this repository:

    python inference.py --config configs/final_config.yaml --checkpoint-dir ../model_release_for_cloud_0.84571_20260606_232402/checkpoints_best_f1 --output reproduced_submission.csv --threshold 0.125 --tta

The threshold 0.125 is the selected threshold for the final ensemble_5fold_best_f1 candidate.

The released final submission is:

    final_submission/01_ensemble_5fold_best_f1_F1_0.994350_pos89_FP1_FN0.csv

With the same test images, sample submission file, configuration, 5 best_f1 checkpoints, threshold=0.125, and TTA enabled, the output should reproduce the released final submission.

## Full training reproduction

Full training requires restoring the original course training data, curated external or pseudo-labeled data, and the local pretrained ConvNeXt safetensors checkpoint referenced by configs/final_config.yaml.

Required training files/directories include:

- train_images/
- train_labels.csv
- outputs/pseudo_stage2/round4_plus_perfect6_round2curated_056119131184pos_114neg_external_train.csv
- pretrained/convnext_small.in12k_ft_in1k_384/model.safetensors

The external training CSV should contain at least these columns:

- id
- path
- label
- sample_weight

The path column must point to image files that are accessible in the current environment.

To rerun training after restoring the required files, use:

    bash scripts/run_round2curated_056119131184pos_114neg_train5fold.sh

To rerun evaluation after restoring the required validation/reference files, use:

    bash scripts/eval_all_round2curated_056119131184pos_114neg_vs_perfect6_1_18_49.sh

## Release notes

This repository is organized for code review and inference reproduction.

The GitHub release intentionally excludes raw images, external image data, pseudo-labeled image data, trained checkpoints, large outputs/logs, private reference labels, API keys, and private credentials.

Complete from-scratch training reproduction depends on restoring the full data assets and may still have small variation due to hardware and randomness.

