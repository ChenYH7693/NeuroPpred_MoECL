import re
import os
import json
import argparse
import torch
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from termcolor import colored
from sklearn.metrics import confusion_matrix

from util.util_process import load_fasta_seq, load_embeddings, load_fasta_ss8, construct_dataset
from model import Model
from util.util_metric import reg_loss, evaluate, caculate_metric

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def load_model(new_model, path_pretrain_model):
    checkpoint = torch.load(path_pretrain_model, map_location=torch.device('cpu'))
    pretrained_dict = checkpoint.get('state_dict', checkpoint) if isinstance(checkpoint, dict) else checkpoint
    if not isinstance(pretrained_dict, dict):
        raise ValueError(f'Unsupported checkpoint format: {path_pretrain_model}')
    new_model_dict = new_model.state_dict()
    unexpected = sorted(set(pretrained_dict) - set(new_model_dict))
    missing = sorted(set(new_model_dict) - set(pretrained_dict))
    shape_mismatch = {
        key: (tuple(value.shape), tuple(new_model_dict[key].shape))
        for key, value in pretrained_dict.items()
        if key in new_model_dict and hasattr(value, 'shape')
        and tuple(value.shape) != tuple(new_model_dict[key].shape)
    }
    if shape_mismatch:
        details = '\n'.join(
            f'  {key}: checkpoint {old_shape} vs model {new_shape}'
            for key, (old_shape, new_shape) in shape_mismatch.items()
        )
        raise RuntimeError(
            f'Checkpoint {path_pretrain_model} is incompatible with the current Model.\n'
            f'{details}\n'
            'This is likely an old PSCG checkpoint. Train the current MoECL model '
            'and pass its MoECL checkpoint to predict.py.'
        )
    compatible = {k: v for k, v in pretrained_dict.items() if k in new_model_dict}
    new_model_dict.update(compatible)
    new_model.load_state_dict(new_model_dict, strict=True)
    if missing:
        print(f'Warning: {len(missing)} model keys were not present in checkpoint.')
    if unexpected:
        print(f'Warning: {len(unexpected)} checkpoint keys were ignored.')
    return new_model


def _save_figure(fig, output_stem):
    """Save publication-friendly raster and vector versions of a figure."""
    os.makedirs(os.path.dirname(output_stem), exist_ok=True)
    fig.savefig(output_stem + '.png', dpi=600, bbox_inches='tight')
    fig.savefig(output_stem + '.svg', bbox_inches='tight')
    fig.savefig(output_stem + '.pdf', bbox_inches='tight')
    plt.close(fig)


def plot_roc_curve(FPR, TPR, AUC, output_stem):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(FPR, TPR, color='#2166AC', lw=2.2, label=f'Ensemble (AUROC = {AUC:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='#8c8c8c', lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('False-positive rate')
    ax.set_ylabel('True-positive rate')
    ax.set_title('Independent test ROC')
    ax.legend(loc='lower right')
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_pr_curve(recall, precision, AUPRC, output_stem):
    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    ax.plot(recall, precision, color='#B2182B', lw=2.2, label=f'Ensemble (AUPRC = {AUPRC:.4f})')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Independent test precision–recall')
    ax.legend(loc='lower left')
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_confusion_matrix(label_real, label_pred, output_stem, threshold):
    cm = confusion_matrix(label_real, label_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    image = ax.imshow(cm, cmap='Blues')
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks([0, 1], ['Non-NP', 'NP'])
    ax.set_yticks([0, 1], ['Non-NP', 'NP'])
    ax.set_xlabel('Predicted label')
    ax.set_ylabel('True label')
    ax.set_title(f'Independent test confusion matrix (threshold = {threshold:.2f})')
    cutoff = cm.max() / 2.0 if cm.max() else 0.0
    for row in range(2):
        for col in range(2):
            ax.text(col, row, str(cm[row, col]), ha='center', va='center',
                    color='white' if cm[row, col] > cutoff else '#222222', fontsize=13)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def plot_metric_summary(performance, output_stem, threshold):
    names = ['ACC', 'AUC', 'AUPRC', 'SE', 'SP', 'F1', 'Precision', 'MCC']
    values = [float(v) for v in performance]
    fig, ax = plt.subplots(figsize=(8.0, 4.4))
    colors = ['#2166AC', '#4393C3', '#67A9CF', '#92C5DE', '#D1E5F0', '#FDDBC7', '#EF8A62', '#B2182B']
    bars = ax.bar(names, values, color=colors, edgecolor='#333333', linewidth=0.5)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel('Score')
    ax.set_title(f'Independent test performance (threshold = {threshold:.2f})')
    ax.grid(axis='y', alpha=0.25, linewidth=0.7)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, f'{value:.4f}',
                ha='center', va='bottom', fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_stem)


def predict(test_iter, path_pretrain_model, sequence_ids, seq_test_dict, threshold=0.5,
            output_dir='Predict'):
    """Run independent prediction with a fixed, validation-selected threshold.

    The threshold must be selected on the training/validation data and then
    passed here unchanged. Never optimize it on the independent test set.
    """
    checkpoint_paths = [path_pretrain_model] if isinstance(path_pretrain_model, str) else list(path_pretrain_model)
    models = []
    for checkpoint_path in checkpoint_paths:
        model = Model(protT5_dim=protT5_dim, max_len=max_len).to(device)
        models.append(load_model(model, checkpoint_path).eval())

    # Average positive-class probabilities across folds. This is the only
    # ensemble operation; the threshold remains fixed from OOF calibration.
    all_probabilities = []
    label_real = []
    with torch.no_grad():
        for x, ss, mask, y in test_iter:
            x, ss, mask = x.to(device), ss.to(device), mask.to(device)
            fold_probabilities = []
            for model in models:
                fold_logits = model(x, ss, mask)
                fold_probabilities.append(torch.softmax(fold_logits, dim=-1)[:, 1])
            all_probabilities.extend(torch.stack(fold_probabilities).mean(dim=0).cpu().tolist())
            label_real.extend(y.tolist())
    pred_prob = all_probabilities
    label_pred = (np.asarray(pred_prob) >= threshold).astype(int).tolist()
    test_performance, test_roc_data, test_prc_data = caculate_metric(
        pred_prob, label_pred, label_real
    )
    # Diagnostic only: compare the fixed OOF threshold with the conventional
    # 0.5 threshold on the same independent probabilities. Do not optimize on
    # the test set or replace the configured threshold with this comparison.
    threshold_05_pred = (np.asarray(pred_prob) >= 0.5).astype(int).tolist()
    threshold_05_performance, _, _ = caculate_metric(pred_prob, threshold_05_pred, label_real)

    # Create a DataFrame containing the prediction results, using sequences_id as the sequence ID.
    sequences = [seq_test_dict[i] for i in sequence_ids]
    results_df = pd.DataFrame({
        'sequence_id': sequence_ids,
        'sequence': sequences,
        'real_label': label_real,
        'pred_prob': pred_prob,
        'pred_label': label_pred
    })

    os.makedirs(output_dir, exist_ok=True)

    output_file = os.path.join(output_dir, 'predict.csv')
    results_df.to_csv(output_file, index=False)
    print(f"The prediction results have been saved to: {output_file}")
    print(f"Decision threshold used for the main result: {threshold:.3f}")
    print(f"Models ensembled: {len(models)}")
    # if abs(float(threshold) - 0.5) < 1e-12:
    #     print('Main independent-test metrics (fixed threshold 0.5): '
    #           f'ACC={test_performance[0]:.4f}, SE={test_performance[3]:.4f}, '
    #           f'SP={test_performance[4]:.4f}, F1={test_performance[5]:.4f}, '
    #           f'MCC={test_performance[7]:.4f}')
    # else:
    #     print('Diagnostic metrics at threshold 0.5 (not the main result): '
    #           f'ACC={threshold_05_performance[0]:.4f}, SE={threshold_05_performance[3]:.4f}, '
    #           f'SP={threshold_05_performance[4]:.4f}, F1={threshold_05_performance[5]:.4f}, '
    #           f'MCC={threshold_05_performance[7]:.4f}')

    # Save numerical metrics and the figures using exactly the same main
    # threshold. AUC/AUPRC are threshold-independent; the other metrics are not.
    metrics_payload = {
        'threshold': float(threshold),
        'models_ensembled': len(models),
        'ACC': float(test_performance[0]),
        'AUC': float(test_performance[1]),
        'AUPRC': float(test_performance[2]),
        'SE': float(test_performance[3]),
        'SP': float(test_performance[4]),
        'F1': float(test_performance[5]),
        'Precision': float(test_performance[6]),
        'MCC': float(test_performance[7]),
    }
    with open(os.path.join(output_dir, 'metrics_threshold_0.5.json' if abs(float(threshold) - 0.5) < 1e-12 else 'metrics.json'), 'w', encoding='utf-8') as handle:
        json.dump(metrics_payload, handle, indent=2)
    figure_dir = os.path.join(output_dir, 'figures')
    plot_roc_curve(test_roc_data[0], test_roc_data[1], test_roc_data[2], os.path.join(figure_dir, 'independent_test_roc'))
    plot_pr_curve(test_prc_data[0], test_prc_data[1], test_prc_data[2], os.path.join(figure_dir, 'independent_test_pr'))
    plot_confusion_matrix(label_real, label_pred, os.path.join(figure_dir, 'independent_test_confusion_matrix'), threshold)
    plot_metric_summary(test_performance, os.path.join(figure_dir, 'independent_test_metrics'), threshold)
    print(f'Figures and metrics saved to: {figure_dir}')

    test_results = '\n' + '=' * 16 + colored(' Independent Test Performance', 'red') + '=' * 16 \
                  + '\n[ACC,\t\tAUC,\t\tAUPRC,\t\tSE,\t\tSP,\t\tF1,\t\tPre,\tMCC]\n' + \
        '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
        test_performance[0], test_performance[1], test_performance[2], test_performance[3],test_performance[4], 
        test_performance[5], test_performance[6], test_performance[7]) + '\n' + '=' * 60

    test_roc_data = np.array(test_roc_data, dtype=object)
    test_prc_data = np.array(test_prc_data, dtype=object)

    return test_results, test_roc_data, test_prc_data


def reorder_by_ids(ids, *dicts):
    """Rearrange multiple dictionaries in the order of the specified IDs"""
    return [{k: d[k] for k in ids} for d in dicts]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate NeuroPpred-MoECL on the independent test set.')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Fixed decision threshold for the main result (default: 0.5).')
    parser.add_argument('--use-oof-threshold', action='store_true',
                        help='Use the threshold selected from training OOF predictions instead of 0.5.')
    args = parser.parse_args()
    max_len = 100
    seq_folder = "dataset/NeuroP_data"
    protT5_folder = 'dataset/NeuroP_feature/ProtT5'
    ss_folder = 'dataset/NeuroP_feature/ss8'

    seq_test_dict, _ = load_fasta_seq(seq_folder, is_training=False)
    protT5_test_dict, labels_test_dict = load_embeddings(protT5_folder, max_length=max_len, is_training=False)
    ss_test_dict, _ , mask_test_dict = load_fasta_ss8(ss_folder, max_length=max_len, is_training=False)

    first_key = list(protT5_test_dict.keys())[0]
    protT5_dim = protT5_test_dict[first_key].shape[1]
    print(f"ProtT5 feature dimension: {protT5_dim}")

    # ID sorting: Positive first, then Negative
    all_ids = list(labels_test_dict.keys())

    pos_ids = sorted([i for i in all_ids if i.startswith("Positive")], key=lambda x: int(re.findall(r"\d+", x)[0]))
    neg_ids = sorted([i for i in all_ids if i.startswith("Negative")], key=lambda x: int(re.findall(r"\d+", x)[0]))
    sequence_ids = pos_ids + neg_ids

    # Unified dictionary order
    protT5_test_dict, ss_test_dict, mask_test_dict, labels_test_dict, seq_test_dict = \
        reorder_by_ids(sequence_ids,protT5_test_dict, ss_test_dict,mask_test_dict,labels_test_dict,seq_test_dict )

    test_iter = construct_dataset(protT5_test_dict, ss_test_dict, mask_test_dict, labels_test_dict, train=False)

    fold_checkpoints = sorted(
        os.path.join('checkpoints', name)
        for name in os.listdir('checkpoints')
        if name.startswith('moecl_fold_') and name.endswith('.pt')
    ) if os.path.isdir('checkpoints') else []
    path_pretrain_model = fold_checkpoints or ['NeuroPpred_MoECL_final.pt']
    threshold_file = 'decision_threshold.json'
    if args.use_oof_threshold and os.path.exists(threshold_file):
        with open(threshold_file, 'r', encoding='utf-8') as handle:
            decision_threshold = float(json.load(handle)['threshold'])
    elif args.use_oof_threshold:
        raise FileNotFoundError(
            f'{threshold_file} not found. Run train.py first to complete CV and final training.'
        )
    else:
        # The independent test set is never used to tune this threshold. The
        # default 0.5 setting is the primary, directly comparable result.
        decision_threshold = args.threshold
    if not 0.0 < decision_threshold < 1.0:
        raise ValueError(f'Threshold must be between 0 and 1, got {decision_threshold}')
    test_result, test_roc_data, test_prc_data = predict(
        test_iter, path_pretrain_model, sequence_ids, seq_test_dict,
        threshold=decision_threshold
    )
    print(test_result)
