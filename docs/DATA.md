# Data Notes

This GitHub release does not include the full image dataset, external augmented images, pseudo-labeled images, or trained model checkpoints.

For inference reproduction, prepare the following files or directories outside the GitHub repository:

- test_images/
- sample_submission.csv
- configs/final_config.yaml
- trained checkpoints downloaded from the cloud-drive model package

The trained checkpoints are provided separately in the model release package:

Baidu Netdisk link: https://pan.baidu.com/s/1ahdqqZy_oSnC91Rde_R0Rw?pwd=quc3

Extraction code: quc3

For full training reproduction, the original course training images and curated external/pseudo data must be restored to the paths referenced by configs/final_config.yaml and the corresponding training CSV files.

The final submitted CSV is included in:

final_submission/01_ensemble_5fold_best_f1_F1_0.994350_pos89_FP1_FN0.csv

The final Kaggle public F1 score is:

0.84571
