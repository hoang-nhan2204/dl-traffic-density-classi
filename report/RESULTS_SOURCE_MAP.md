# Results Source Map

This map records the provenance of every quantitative result used in `RESULTS_AND_CONCLUSION_DRAFT.md`. “Final clean” refers to `data/transfer_final_manifests/`, not to the original ZIP folder split.

| Report result/table/figure | Model/run | Split | Source file/path | Notes |
|---|---|---|---|---|
| Final-manifest sizes and Category-A exclusions | Shared data contract | Train/validation/test | `data/transfer_final_manifests/summary.csv` | 2,691 / 576 / 574 after removing 16 / 4 / 7 records. |
| Class supports for final test | Shared data contract | Final clean test | `data/transfer_final_manifests/test.csv` | Empty 191, Low 159, Medium 118, High 67, Traffic Jam 39. |
| Preprocessing and loader contract | Transfer pipeline | All final splits | `src/transfer_data.py` | RGB conversion, 224 letterbox, padding `(124,116,103)`, one-hot labels. |
| Augmentation ablation | EfficientNetB0 A0/A1/A2, F1, weighted CE, seed 42 | Validation | `outputs/transfer_learning/experiment_registry.csv`; each run’s `history.json` | Accuracy is taken at best validation Macro-F1 epoch. |
| Fine-tuning ablation | EfficientNetB0 A0 F0/F1/F2, weighted CE, seed 42 | Validation | `outputs/transfer_learning/experiment_registry.csv`; each run’s `history.json` | F0 frozen; F1/F2 unfreeze 25%/50% by `scripts/run_transfer_experiments.py`. |
| Loss ablation | EfficientNetB0 A0/F2, CE/weighted/smooth, seed 42 | Validation | `outputs/transfer_learning/experiment_registry.csv`; each run’s `history.json` | CE selected by validation Macro-F1. |
| Transfer development selection | EfficientNetB0 A0/F2/CE, seed 42 | Validation | `outputs/transfer_learning/efficientnetb0_A0_F2_ce_s42/completed.json` | Best validation Macro-F1 = 0.885958. |
| Final transfer-backbone comparison | MobileNetV2, EfficientNetB0, ResNet50; A0/F2/CE, seed 42 | Final clean test | `outputs/transfer_learning/final_test_comparison.csv` | All three use 574 test images; parameter counts are saved in this file. |
| Selected model test metrics and per-class table | EfficientNetB0 A0/F2/CE, seed 42 | Final clean test | `outputs/transfer_learning/efficientnetb0_A0_F2_ce_s42/test_metrics.json` | Primary final result: Macro-F1 = 0.831337. |
| Figure 4.1 / confusion analysis | EfficientNetB0 A0/F2/CE, seed 42 | Final clean test | `outputs/transfer_learning/efficientnetb0_A0_F2_ce_s42/test_metrics.json`; generated `report/figures/efficientnetb0_a0_f2_ce_s42_test_confusion_matrix.png` | Figure is a direct rendering of the saved 5 x 5 matrix. |
| Ordinal analysis | EfficientNetB0 A0/F2/CE, seed 42 | Final clean test | `outputs/transfer_learning/efficientnetb0_A0_F2_ce_s42/test_metrics.json` | Ordinal MAE = 0.177700; within-one accuracy = 0.989547. |
| Figure 4.2 / training behaviour | EfficientNetB0 A0/F2/CE, seed 42 | Train/validation | `outputs/transfer_learning/efficientnetb0_A0_F2_ce_s42/history.json`; generated `report/figures/efficientnetb0_a0_f2_ce_s42_learning_curves.png` | Existing history only; no retraining. |
| Multi-seed validation and test summary | EfficientNetB0 A0/F2/CE, seeds 42/43/44 | Validation and final clean test | Each seed’s `completed.json` and `test_metrics.json` under `outputs/transfer_learning/` | Validation mean +/- sample SD is reported separately from test values. Test confusion matrices differ, although each has 480/574 correct. |
| Complex CNN selected configuration | Deep CNN baseline | Validation | `outputs/complex_cnn/experiment_registry.csv`; `outputs/complex_cnn/selected_config.json` | Selected by runner’s validation Macro-F1 / accuracy / loss ordering. |
| Complex CNN final test row | Deep CNN baseline | Final clean test | `outputs/complex_cnn/test_metrics.json` | 574-image test; Macro-F1 = 0.766877; no ordinal values saved. |
| Complex CNN curves | Deep CNN baseline | Train/validation | `outputs/complex_cnn/exp1_baseline/curves.png`; `history.csv` | Referenced qualitatively only. |
| Legacy Simple CNN context row | Simple CNN A1 + weighted CE, seeds 42/43/44 | 587-sample legacy test | `outputs/summary/final_comparison.csv`; `outputs/simple_A1_weighted_{42,43,44}/metrics.json` | Not comparable with the final 574-image protocol: no saved manifest reference or parameter count. |
| Near-duplicate limitation | Dataset audit | Cross-split candidates | `NEAR_DUPLICATE_VERIFICATION.md`; `outputs/near_duplicate_audit/near_duplicate_pairs.csv` | Category A was handled for final manifests; Category B remains review-only. |
| Grad-CAM status | All models | N/A | Repository-wide artifact search | No Grad-CAM/saliency/failure-case artifacts found. |
