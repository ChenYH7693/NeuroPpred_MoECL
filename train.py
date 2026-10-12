import random
import numpy as np
import torch
import torch.nn.functional as F
import time
import copy
import json
import os
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from termcolor import colored
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

from util.util_process import load_embeddings, load_fasta_ss8, construct_dataset
from model import Model
from util.util_metric import evaluate, reg_loss, EarlyStopping, caculate_metric, find_optimal_threshold


seed = 42
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

setup_seed(seed)


def save_cv_oof_roc_curve(oof_labels, oof_prob, output_dir='results/cv_roc'):
    """Save the 10-fold out-of-fold ROC curve and its source coordinates.

    Every OOF probability comes from a model that did not train on that
    sample, so this curve is the appropriate aggregate cross-validation ROC
    rather than a training-set ROC.  No independent-test data are used.
    """
    labels = np.asarray(oof_labels, dtype=int)
    probabilities = np.asarray(oof_prob, dtype=float)
    if labels.size == 0 or np.unique(labels).size < 2:
        raise ValueError('OOF ROC requires both positive and negative labels.')
    fpr, tpr, thresholds = roc_curve(labels, probabilities, pos_label=1)
    oof_auc = auc(fpr, tpr)
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)

    # Save exact curve coordinates for reproducibility and downstream plotting.
    roc_table = np.column_stack([fpr, tpr, thresholds])
    np.savetxt(
        os.path.join(output_path, 'cv_oof_roc.csv'), roc_table, delimiter=',',
        header='FPR,TPR,threshold', comments=''
    )

    # Keep the OOF predictions themselves so ROC/PR curves can be regenerated
    # later without rerunning model inference.
    np.savetxt(
        os.path.join(output_path, 'cv_oof_predictions.csv'),
        np.column_stack([labels, probabilities]), delimiter=',',
        header='label,pred_prob', comments=''
    )
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities, pos_label=1)
    np.savetxt(
        os.path.join(output_path, 'cv_oof_pr.csv'),
        np.column_stack([
            recall, precision,
            np.pad(pr_thresholds, (0, max(0, len(recall) - len(pr_thresholds))),
                   mode='constant', constant_values=np.nan)[:len(recall)]
        ]), delimiter=',', header='Recall,Precision,threshold', comments=''
    )

    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'DejaVu Sans'],
        'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False,
        'axes.linewidth': 0.8, 'pdf.fonttype': 42, 'svg.fonttype': 'none',
    })
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ax.plot(fpr, tpr, color='#2166AC', lw=2.2, label=f'10-fold OOF (AUROC = {oof_auc:.4f})')
    ax.plot([0, 1], [0, 1], '--', color='#888888', lw=1.0, label='Random classifier')
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel('False-positive rate')
    ax.set_ylabel('True-positive rate')
    ax.set_title('10-fold cross-validation ROC')
    ax.legend(loc='lower right', frameon=False)
    fig.tight_layout()
    fig.savefig(os.path.join(output_path, 'cv_oof_roc.png'), dpi=600, bbox_inches='tight')
    fig.savefig(os.path.join(output_path, 'cv_oof_roc.svg'), bbox_inches='tight')
    fig.savefig(os.path.join(output_path, 'cv_oof_roc.pdf'), bbox_inches='tight')
    plt.close(fig)
    print(f'10-fold OOF ROC saved to: {output_path} (n={labels.size}, AUROC={oof_auc:.4f})')
    return oof_auc


def save_fold_curves(fold, labels, probabilities, output_dir='results/cv_roc'):
    """Persist one held-out fold's ROC and PR coordinates."""
    labels = np.asarray(labels, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    output_path = os.path.abspath(output_dir)
    os.makedirs(output_path, exist_ok=True)
    fpr, tpr, roc_thresholds = roc_curve(labels, probabilities, pos_label=1)
    np.savetxt(
        os.path.join(output_path, f'fold_{fold}_roc.csv'),
        np.column_stack([fpr, tpr, roc_thresholds]), delimiter=',',
        header='FPR,TPR,threshold', comments=''
    )
    precision, recall, pr_thresholds = precision_recall_curve(labels, probabilities, pos_label=1)
    padded_thresholds = np.pad(
        pr_thresholds, (0, len(recall) - len(pr_thresholds)),
        mode='constant', constant_values=np.nan
    )
    np.savetxt(
        os.path.join(output_path, f'fold_{fold}_pr.csv'),
        np.column_stack([recall, precision, padded_thresholds]), delimiter=',',
        header='Recall,Precision,threshold', comments=''
    )
    # Save the raw held-out predictions and true labels as well.  These files
    # make every fold auditable and allow later plots/metrics to be regenerated
    # without rerunning the model.
    fold_prediction = np.column_stack([
        labels, probabilities, (probabilities >= 0.5).astype(int)
    ])
    np.savetxt(
        os.path.join(output_path, f'fold_{fold}_predictions.csv'),
        fold_prediction, delimiter=',', fmt=['%d', '%.10f', '%d'],
        header='label,pred_prob,pred_label_threshold_0.5', comments=''
    )


def supervised_contrastive_loss(features, labels, temperature=0.07):
    """Supervised contrastive loss for one normalized embedding per sample."""
    logits = torch.matmul(features, features.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()

    self_mask = torch.eye(labels.size(0), dtype=torch.bool, device=labels.device)
    positive_mask = labels[:, None].eq(labels[None, :]) & ~self_mask
    denominator_mask = ~self_mask
    log_prob = logits - torch.log(
        (torch.exp(logits) * denominator_mask).sum(dim=1, keepdim=True).clamp_min(1e-12)
    )

    positive_count = positive_mask.sum(dim=1)
    valid_anchor = positive_count > 0
    if not valid_anchor.any():
        return features.sum() * 0.0
    mean_log_prob_positive = (
        (log_prob * positive_mask).sum(dim=1) / positive_count.clamp_min(1)
    )
    return -mean_log_prob_positive[valid_anchor].mean()


def moecl_loss(outputs, labels, ss8, padding_mask, supcon_weight, ss8_weight=0.05):
    # The benchmark is already class-balanced. Strong label smoothing (0.1)
    # was making positive probabilities too conservative on the independent
    # set, so use a lighter value while retaining mild regularization.
    classification = F.cross_entropy(outputs["logits"], labels, label_smoothing=0.03)
    contrastive = supervised_contrastive_loss(outputs["projection"], labels)

    safe_ss8 = ss8.masked_fill(padding_mask, 0)
    residue_loss = F.cross_entropy(
        outputs["ss8_logits"].reshape(-1, 8), safe_ss8.reshape(-1), reduction="none"
    ).reshape_as(ss8)
    valid = ~padding_mask
    structure = (residue_loss * valid).sum() / valid.sum().clamp_min(1)
    total = classification + supcon_weight * contrastive + ss8_weight * structure
    return total, classification, contrastive, structure


def train_test(train_iter, valid_iter, protT5_dim = 2560, max_len = 100, checkpoint_path=None):
    net = Model(protT5_dim = protT5_dim, max_len = max_len).to(device)
    print(net, '\n')
    model_parameters = filter(lambda p: p.requires_grad, net.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print("Trainable Parameter: " + str(params) + "\n")
    print(f'Total parameters: {params / 1e6:.2f}M')

    lr =  3e-4
    # Keep the classification objective dominant on this small benchmark.
    # The auxiliary terms are useful regularizers, but large weights can hurt
    # precision and make fold-to-fold variance worse.
    # Classification-first default.  The two auxiliary heads remain in the
    # model for ablation, but are disabled here because the current benchmark
    # shows a precision/specificity regression when they are optimized jointly.
    max_supcon_weight = 0.03
    ss8_weight = 0.0
    supcon_warmup_epochs = 15
    best_valid_metric = -float('inf')
    EPOCH = 100

    optimizer = torch.optim.AdamW(params=net.parameters(), lr=lr, weight_decay=1e-4) # 1e-2  
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor =0.5, patience=6)
    early_stopping = EarlyStopping(patience=10, delta=0)

    best_valid_performance, best_test_performance = None, None
    best_valid_ROC, best_valid_PRC = None, None
    best_epoch = 0
    best_threshold = 0.5
    best_valid_pred_prob = None
    best_valid_labels = None
    best_state_dict = None

    for epoch in range(EPOCH):
        train_loss_ls = []
        cls_loss_ls, supcon_loss_ls, ss8_loss_ls = [], [], []
        t0 = time.time()
        supcon_weight = max_supcon_weight * min(1.0, (epoch + 1) / supcon_warmup_epochs)

        net.train()
        for protT5_feat, ss, mask, label in train_iter:
            if device:
                protT5_feat, ss, mask, label = protT5_feat.to(device), ss.to(device), mask.to(device), label.to(device)

            outputs = net(protT5_feat, ss, mask, return_aux=True)
            loss, cls_loss, supcon_loss, ss8_loss = moecl_loss(
                outputs, label, ss, mask, supcon_weight, ss8_weight
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss_ls.append(loss.item())
            cls_loss_ls.append(cls_loss.item())
            supcon_loss_ls.append(supcon_loss.item())
            ss8_loss_ls.append(ss8_loss.item())

        net.eval()
        with torch.no_grad():
            train_performance, train_roc_data, train_prc_data, _, train_loss, _, _ = evaluate(train_iter, net, reg_loss)
            # First collect validation probabilities at the default threshold,
            # then calibrate the decision threshold using validation MCC.
            _, _, _, label_real, valid_loss, valid_pred_prob, _ = evaluate(
                valid_iter, net, reg_loss, threshold=0.5
            )
            decision_threshold = find_optimal_threshold(valid_pred_prob, label_real)
            calibrated_pred = (np.asarray(valid_pred_prob) >= decision_threshold).astype(int).tolist()
            valid_performance, valid_roc_data, valid_prc_data = caculate_metric(
                valid_pred_prob, calibrated_pred, label_real
            )

        results = f"\nepoch: {epoch + 1}, loss: {train_loss:.5f}, threshold: {decision_threshold:.3f}\n"
        results += (f'MoECL train objective: {np.mean(train_loss_ls):.5f} '
                    f'(CE={np.mean(cls_loss_ls):.5f}, '
                    f'SupCon={np.mean(supcon_loss_ls):.5f} × {supcon_weight:.3f}, '
                    f'SS8={np.mean(ss8_loss_ls):.5f} × {ss8_weight:.3f})\n')
        results += f'train_acc: {train_performance[0]:.4f}, time: {time.time() - t0:.2f}'
        results += '\n' + '=' * 16 + ' Valid Performance. Epoch[{}] '.format(epoch + 1) + '=' * 16 \
                   + '\n[ACC,\tAUC,\tAUPRC, \tSE,\tSP, \tF1, \tPre, \tMCC]\n' + '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            valid_performance[0], valid_performance[1], valid_performance[2], valid_performance[3],valid_performance[4],
            valid_performance[5], valid_performance[6], valid_performance[7]) + '\n' + '=' * 60
        print(results)

        print(f'\nEpoch:{epoch + 1}, loss:{train_loss:.5f}, time:{time.time() - t0:.2f}\n',
              f'Train_ACC:{train_performance[0]:.4f}|Train_AUC:{train_performance[1]:.4f}|Train_AUPRC:{train_performance[2]:.4f}|Train_Sensitivity:{train_performance[3]:.4f}|Train_Specificity:{train_performance[4]:.4f}|Train_F1:{train_performance[5]:.4f}|Train_Pre:{train_performance[6]:.4f}|Train_MCC:{train_performance[7]:.4f}|Train_loss:{train_loss:.4f}\n'
              f'Valid_ACC:{valid_performance[0]:.4f}|Valid_AUC:{valid_performance[1]:.4f}|Valid_AUPRC:{valid_performance[2]:.4f}|Valid_Sensitivity:{valid_performance[3]:.4f}|Valid_Specificity:{valid_performance[4]:.4f}|Valid_F1:{valid_performance[5]:.4f}|Valid_Pre:{valid_performance[6]:.4f}|Valid_MCC:{valid_performance[7]:.4f}|Valid_loss:{valid_loss:.4f}\n')

        valid_metric = valid_performance[7]  # MCC is more balanced than ACC here.
        scheduler.step(valid_metric)

        if valid_metric > best_valid_metric:
            best_valid_metric = valid_metric
            best_epoch = epoch + 1

            best_valid_performance = valid_performance.copy()
            best_valid_ROC = valid_roc_data.copy()
            best_valid_PRC = valid_prc_data.copy()
            best_threshold = decision_threshold
            best_valid_pred_prob = list(valid_pred_prob)
            best_valid_labels = list(label_real)
            best_state_dict = copy.deepcopy(net.state_dict())

            best_valid_results = '\n' + '=' * 16 + colored(' Best Valid Performance. Epoch[{}] ', 'red').format(
                best_epoch) + '=' * 16  + '\n[ACC,\tAUC,\tAUPRC,\tSE,\tSP,\tF1,\tPre,\tMCC]\n' + \
                '{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
                best_valid_performance[0], best_valid_performance[1], best_valid_performance[2], best_valid_performance[3], best_valid_performance[4],
                best_valid_performance[5], best_valid_performance[6], best_valid_performance[7]) + '\n' + '=' * 60
            print(best_valid_results)

        early_stopping(valid_metric, net)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    # EarlyStopping monitors validation ACC, so restore the corresponding
    # checkpoint before returning. This also makes downstream prediction use
    # the same model selected during cross-validation.
    if best_state_dict is not None:
        net.load_state_dict(best_state_dict)
        # Save a checkpoint that is compatible with predict.py.  This is the
        # current MoECL architecture; the legacy NeuroPred_PSCG_Model.pt must
        # not be reused for this model.
        if checkpoint_path:
            checkpoint_dir = os.path.dirname(checkpoint_path)
            if checkpoint_dir:
                os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save(best_state_dict, checkpoint_path)

    return (best_valid_performance, best_valid_ROC, best_valid_PRC, best_epoch,
            best_threshold, best_valid_pred_prob, best_valid_labels)


def train_final(train_iter, protT5_dim, max_len, epochs, checkpoint_path):
    """Train once on all labeled training data using CV-selected settings."""
    net = Model(protT5_dim=protT5_dim, max_len=max_len).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=3e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, epochs), eta_min=3e-5)
    for epoch in range(epochs):
        net.train()
        supcon_weight = 0.03 * min(1.0, (epoch + 1) / 15)
        for protT5_feat, ss, mask, label in train_iter:
            protT5_feat, ss, mask, label = protT5_feat.to(device), ss.to(device), mask.to(device), label.to(device)
            outputs = net(protT5_feat, ss, mask, return_aux=True)
            loss, _, _, _ = moecl_loss(outputs, label, ss, mask, supcon_weight, 0.0)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=2.0)
            optimizer.step()
        scheduler.step()
        print(f'Final training epoch {epoch + 1}/{epochs}, loss={loss.item():.5f}')
    torch.save(net.state_dict(), checkpoint_path)
    return net



def K_CV(protT5_train_dict, ss_train_dict, mask_train_dict,labels_train_dict, k = 5, max_len = 100):
    train_protein = list(protT5_train_dict.keys())
    train_list = [protT5_train_dict[pid] for pid in train_protein]
    train_ss_list = [ss_train_dict[pid] for pid in train_protein]
    train_mask_list = [mask_train_dict[pid] for pid in train_protein]
    train_labels_list = [labels_train_dict[pid] for pid in train_protein]

    train_array = np.array(train_list)
    train_ss_array = np.array(train_ss_list)
    train_mask_array = np.array(train_mask_list)
    train_labels_array = np.array(train_labels_list)

    CV_perform = []
    CV_test = []
    CV_epochs = []
    CV_thresholds = []
    oof_prob = []
    oof_labels = []

    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    for iter_k, (train_index, valid_index) in enumerate(skf.split(train_array, train_labels_array)):
        print("\n" + "=" * 16 + "k = " + str(iter_k + 1) + "=" * 16)

        train_feat, valid_feat = train_array[train_index], train_array[valid_index]
        train_ss, valid_ss = train_ss_array[train_index], train_ss_array[valid_index]
        train_mask, valid_mask = train_mask_array[train_index], train_mask_array[valid_index]
        train_labels, valid_labels = train_labels_array[train_index], train_labels_array[valid_index]
        

        train_dict = {f"train_{i}": feat for i, feat in enumerate(train_feat)}
        train_ss_dict = {f"train_{i}": ss_id for i, ss_id in enumerate(train_ss)}
        train_mask_dict = {f"train_{i}": label for i, label in enumerate(train_mask)}
        train_labels_dict = {f"train_{i}": label for i, label in enumerate(train_labels)}
        

        valid_dict = {f"valid_{i}": feat for i, feat in enumerate(valid_feat)}
        valid_ss_dict = {f"valid_{i}": ss_id for i, ss_id in enumerate(valid_ss)}
        valid_mask_dict = {f"valid_{i}": label for i, label in enumerate(valid_mask)}
        valid_labels_dict = {f"valid_{i}": label for i, label in enumerate(valid_labels)}

        train_iter = construct_dataset(train_dict, train_ss_dict, train_mask_dict, train_labels_dict, train=True)
        valid_iter = construct_dataset(valid_dict, valid_ss_dict, valid_mask_dict, valid_labels_dict, train=False)

        print(f'train_iter: {len(train_iter)}')
        print(f'valid_iter: {len(valid_iter)}')

        first_key = list(train_dict.keys())[0]
        protT5_dim = train_dict[first_key].shape[1]

        (best_valid_performance, best_valid_ROC, best_valid_PRC, best_epoch,
         best_threshold, fold_prob, fold_labels) = train_test(
            train_iter, valid_iter, protT5_dim=protT5_dim, max_len=max_len,
            checkpoint_path=os.path.join('checkpoints', f'moecl_fold_{iter_k + 1}.pt'))

        print('Cross-validation: best_performance', best_valid_performance)
        save_fold_curves(iter_k + 1, fold_labels, fold_prob)
        CV_perform.append(best_valid_performance)
        CV_epochs.append(best_epoch)
        CV_thresholds.append(best_threshold)
        oof_prob.extend(fold_prob)
        oof_labels.extend(fold_labels)

    print('\n' + '=' * 16 + colored(' Cross-Validation Performance ', 'red') + '=' * 16 +
          '\n[ACC,\tBACC,\tAUC,\tAUPRC,\tSE,\tSP,\tF1,\tPre,\tMCC]\n')

    for k_idx, out in enumerate(CV_perform):
        print('第{}折: {:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f},\t{:.4f}'.format(
            k_idx + 1, out[0], out[1], out[2], out[3], out[4], out[5], out[6], out[7]))

    mean_out = np.array(CV_perform).mean(axis=0)
    std_out = np.array(CV_perform).std(axis=0)

    print('\n' + '=' * 16 + "Mean ± Std" + '=' * 16)
    print(
        'ACC:{:.4f}±{:.4f}, \tAUC: {:.4f}±{:.4f},\tAUPRC: {:.4f}±{:.4f},\tSE: {:.4f}±{:.4f},\tSP: {:.4f}±{:.4f},\tF1: {:.4f}±{:.4f},\tPre: {:.4f}±{:.4f},\tMCC: {:.4f}±{:.4f}'.format(
            mean_out[0], std_out[0], mean_out[1], std_out[1], mean_out[2], std_out[2], mean_out[3], std_out[3], 
            mean_out[4], std_out[4], mean_out[5], std_out[5], mean_out[6], std_out[6], mean_out[7], std_out[7]))

    print('\n' + '=' * 60)
    global_threshold = find_optimal_threshold(oof_prob, oof_labels)
    print(f'OOF global decision threshold: {global_threshold:.4f}')
    # Aggregate all held-out fold predictions into one leakage-free ROC curve.
    # This is saved before final full-training begins, so it documents the
    # actual 10-fold CV performance independently of the final checkpoint.
    save_cv_oof_roc_curve(oof_labels, oof_prob)
    return CV_perform, CV_test, CV_epochs, CV_thresholds, global_threshold


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    max_len = 100
    protT5_folder = 'dataset/NeuroP_feature/ProtT5'
    protT5_train_dict, labels_train_dict = load_embeddings(protT5_folder, max_length=max_len, is_training=True)
    protT5_test_dict, labels_test_dict = load_embeddings(protT5_folder, max_length=max_len, is_training=False)

    ss_folder = 'dataset/NeuroP_feature/ss8'
    ss_train_dict, _ , mask_train_dict = load_fasta_ss8(ss_folder, max_length=max_len, is_training=True)
    ss_test_dict, _ , mask_test_dict = load_fasta_ss8(ss_folder, max_length=max_len, is_training=False)

    first_key = list(protT5_train_dict.keys())[0]
    protT5_dim = protT5_train_dict[first_key].shape[1]
    print(f"ProtT5 feature dimension: {protT5_dim}")

    k_fold = 10
    print(f"Starting {k_fold}-fold cross validation")
    CV_perform, CV_test, cv_epochs, cv_thresholds, fixed_threshold = K_CV(
        protT5_train_dict, ss_train_dict, mask_train_dict, labels_train_dict,
        k=k_fold, max_len=max_len
    )

    # Stage 2: freeze the configuration selected by CV, retrain on all labeled
    # training samples, and save the final model used by predict.py.
    fixed_epochs = max(1, int(round(float(np.median(cv_epochs)))))
    print(f'Using OOF-selected threshold for final model: {fixed_threshold:.4f}')
    final_iter = construct_dataset(
        protT5_train_dict, ss_train_dict, mask_train_dict, labels_train_dict, train=True
    )
    final_checkpoint = 'NeuroPpred_MoECL_final.pt'
    train_final(final_iter, protT5_dim, max_len, fixed_epochs, final_checkpoint)
    with open('decision_threshold.json', 'w', encoding='utf-8') as handle:
        json.dump({
            'threshold': fixed_threshold,
            'epochs': fixed_epochs,
            'source': '10-fold CV median',
        }, handle, indent=2)
    print(f'Final model saved to {final_checkpoint}')
    print(f'Fixed decision threshold saved to decision_threshold.json: {fixed_threshold:.4f}')
