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

[MODEL_CLOUD_DRIVE_LINK_TO_BE_FILLED]

The model package contains 10 checkpoints:

- checkpoints_best_f1/fold0_best_f1.pth to fold4_best_f1.pth
- checkpoints_best_loss/fold0_best_loss.pth to fold4_best_loss.pth

For reproducing the final submitted CSV, use the 5 best_f1 checkpoints.

After downloading and extracting the model package, create an alias directory in this repository:

    mkdir -p checkpoints_final_best_f1_alias
    cp /path/to/model_release_for_cloud_0.84571_20260606_232402/checkpoints_best_f1/fold*_best_f1.pth checkpoints_final_best_f1_alias/

The inference script expects fold checkpoint files inside one checkpoint directory.

## Inference reproduction

Prepare the required test data files according to configs/final_config.yaml and docs/DATA.md.

Then run:

    python inference.py --config configs/final_config.yaml --checkpoint-dir checkpoints_final_best_f1_alias --output reproduced_submission.csv --tta

The released final submission is:

    final_submission/01_ensemble_5fold_best_f1_F1_0.994350_pos89_FP1_FN0.csv

If the same test images, sample submission file, configuration, and 5 best_f1 checkpoints are used, the inference output should reproduce the final submission.

## Full training reproduction

Full training requires restoring the original course training images and the curated external or pseudo-labeled data used by configs/final_config.yaml.

The GitHub release intentionally excludes:

- raw train images
- raw test images
- external or pseudo image data
- trained checkpoints
- large outputs and logs
- reference label files used only for internal validation

To rerun training after restoring the required data paths, use:

    bash scripts/run_round2curated_056119131184pos_114neg_train5fold.sh

To rerun evaluation after restoring the required validation/reference files, use:

    bash scripts/eval_all_round2curated_056119131184pos_114neg_vs_perfect6_1_18_49.sh

## Notes

This release is organized for code review and inference reproduction. The final model weights are provided separately through the cloud-drive model package.

