# NeuroMESH

NeuroMESH is a multimodal model for neuropeptide prediction. This repository contains the implementation published as **NeuroPpred-MoECL**. It combines residue-level ProtT5 representations with eight-state secondary-structure (SS8) features and uses three complementary experts with sample-level soft routing.

## Model overview

ProtT5 and SS8 features are aligned by residue position and processed by three expert branches:

- **Local-motif expert:** parallel depthwise-separable convolutions with kernel sizes 3, 5, and 7 capture short-range sequence patterns.
- **Global-context expert:** a bidirectional selective state-space recurrence models dependencies across the peptide.
- **Structure-aware expert:** SS8 embeddings modulate ProtT5 features.

A learned router combines the expert outputs for each peptide. Masked attention pooling produces a peptide-level representation for binary classification. Training combines label-smoothed cross-entropy with a supervised contrastive objective. The model also defines an auxiliary SS8 prediction head, although its loss weight is set to zero in the current training script.

## Repository contents

```text
.
├── model.py                 # NeuroPpred-MoECL architecture
├── train.py                 # 10-fold cross-validation and final training
├── predict.py               # Independent-set evaluation and prediction
├── util/
│   ├── util_process.py       # Feature loading and batching
│   └── util_metric.py        # Metrics and threshold selection
└── dataset/
    └── NeuroP_data/          # FASTA/CSV sequence data and split files
```

The ProtT5 embedding arrays and SS8 FASTA feature files expected by the scripts are not included in this repository. The scripts expect them under `dataset/NeuroP_feature/ProtT5` and `dataset/NeuroP_feature/ss8`, respectively. See [Input features](#input-features) before running training or prediction.

## Requirements

The code imports Python packages from PyTorch, NumPy, pandas, scikit-learn, Matplotlib, Biopython, and `termcolor`. Use a Python environment with compatible versions of these packages and install a PyTorch build suited to your CPU or CUDA setup. The repository does not currently provide a pinned `requirements.txt` or environment file.

## Input features

The current training and prediction scripts use precomputed residue-level features. ProtT5 feature generation and SS8 prediction are not implemented in the included scripts, so prepare these files before running either script.

### ProtT5 embeddings

Place one `.npy` array per sequence in the following directories:

```text
dataset/NeuroP_feature/ProtT5/
├── train_pos_ProtT5/
├── train_neg_ProtT5/
├── test_pos_ProtT5/
└── test_neg_ProtT5/
```

Each array should contain residue-level ProtT5-XL-UniRef50 representations, with shape `(sequence_length, embedding_dimension)`. Use the FASTA record identifier as the `.npy` filename stem so it matches the corresponding SS8 identifier. The loader truncates or zero-pads arrays to 100 residues.

### SS8 sequences

Place the corresponding structure-state sequences in:

```text
dataset/NeuroP_feature/ss8/
├── train_pos_ss8.fasta
├── train_neg_ss8.fasta
├── test_pos_ss8.fasta
└── test_neg_ss8.fasta
```

Use the same record identifiers as in the ProtT5 arrays. The loader maps `H`, `E`, `C`, `S`, `T`, `G`, `I`, and `B` to the eight SS8 states. `X` is treated as unknown or padding. Sequences are truncated or padded to 100 positions, and padded positions are masked during model processing.

### Labels and splits

The loaders assign label `1` to positive neuropeptides and label `0` to negative non-neuropeptides based on the positive/negative directory and FASTA filenames. Keep the training and independent test partitions separate. Do not use the independent test set to select a decision threshold or tune model settings.

## Training

From the repository root, after preparing the feature directories above, run:

```bash
python train.py
```

The script performs 10-fold stratified cross-validation on the training partition, saves fold checkpoints under `checkpoints/`, derives a decision threshold from out-of-fold predictions, and trains a final model using the training partition. Expected outputs include:

```text
NeuroPpred_MoECL_final.pt
decision_threshold.json
checkpoints/moecl_fold_*.pt
results/
```

Training settings such as maximum sequence length, number of folds, and optimization parameters are currently defined in `train.py`, rather than exposed as command-line options.

## Prediction and evaluation

After preparing the test ProtT5 and SS8 features and placing trained checkpoints in the expected locations, run:

```bash
python predict.py
```

By default, prediction uses a threshold of `0.5`. To use the threshold selected from training out-of-fold predictions, run:



Alternatively, provide a fixed threshold selected using training or validation data:

```bash
python predict.py --threshold 0.47
```

When fold checkpoints are present in `checkpoints/`, the script averages their positive-class probabilities. Otherwise, it falls back to `NeuroPpred_MoECL_final.pt`. Results are written to `Predict/`, including `predict.csv`, metrics, and ROC, precision-recall, confusion-matrix, and metric-summary figures.


## Citation

If you use this code or model, please cite the associated manuscript. Add the final bibliographic details or DOI here when available.

## License

No license file is currently included. Unless a license is added by the authors, reuse and redistribution permissions are not specified.
