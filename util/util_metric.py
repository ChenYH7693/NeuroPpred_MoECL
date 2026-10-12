import os
import pandas as pd
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def caculate_metric(pred_prob, label_pred, label_real):
    ACC = accuracy_score(label_real, label_pred)
    F1 = f1_score(label_real, label_pred)
    Pre = precision_score(label_real, label_pred, pos_label=1)  # 二分类正类精度 Precision
    MCC = matthews_corrcoef(label_real, label_pred)

    tn, fp, fn, tp = confusion_matrix(label_real, label_pred, labels=[0, 1]).ravel()
    Sensitivity = tp / (tp + fn)  # recall
    Specificity = tn / (tn + fp)

    # ROC and AUC
    FPR, TPR, thresholds = roc_curve(label_real, pred_prob, pos_label=1)
    AUC = auc(FPR, TPR)

    # PRC and AP
    precision, recall, thresholds = precision_recall_curve(label_real, pred_prob, pos_label=1)
    AP = average_precision_score(label_real, pred_prob, average='macro', pos_label=1, sample_weight=None)
    AUPRC = auc(recall, precision)

    performance = [ACC, AUC, AUPRC, Sensitivity, Specificity,  F1, Pre, MCC]

    roc_data = [FPR, TPR, AUC]
    prc_data = [recall, precision, AUPRC, AP]
    return performance, roc_data, prc_data


def find_optimal_threshold(pred_prob, label_real, low=0.30, high=0.70, steps=161):
    """Find a validation-only threshold that maximizes MCC.

    AUROC is threshold independent, whereas ACC/F1/MCC are not. Searching the
    threshold on the validation fold is useful when the probability calibration
    is slightly biased toward the positive class.
    """
    pred_prob = np.asarray(pred_prob, dtype=float)
    label_real = np.asarray(label_real, dtype=int)
    best_threshold, best_score = 0.5, -np.inf
    for threshold in np.linspace(low, high, steps):
        label_pred = (pred_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(label_real, label_pred, labels=[0, 1]).ravel()
        denominator = np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        mcc = ((tp * tn - fp * fn) / denominator) if denominator > 0 else 0.0
        # Prefer thresholds close to 0.5 when MCC is effectively tied.
        if mcc > best_score + 1e-8 or (abs(mcc - best_score) <= 1e-8 and abs(threshold - 0.5) < abs(best_threshold - 0.5)):
            best_score = mcc
            best_threshold = float(threshold)
    return best_threshold



def evaluate(data_iter, net, criterion, threshold=0.5):
    pred_prob = []
    label_pred = []
    label_real = []
    total_loss = 0.0  
    num_batches = 0  

    for x, ss, mask, y in data_iter:
        x, ss, mask, y = x.to(device), ss.to(device), mask.to(device), y.to(device)
        outputs = net(x, ss, mask).to(device)

        loss = criterion(net,outputs,y)
        total_loss += loss.item()
        num_batches += 1

        outputs = F.softmax(outputs, dim=-1)
        pred_prob_positive = outputs[:, 1]
        pred_prob = pred_prob + pred_prob_positive.tolist()
        label_real = label_real + y.tolist()

    label_pred = (np.asarray(pred_prob) >= threshold).astype(int).tolist()
    performance, roc_data, prc_data = caculate_metric(pred_prob, label_pred, label_real)
    average_loss = total_loss / num_batches
    return performance, roc_data, prc_data,label_real, average_loss, pred_prob, label_pred



def reg_loss(net, output, label):
    criterion = nn.CrossEntropyLoss(reduction='sum',label_smoothing=0.1) # 
    l2_lambda = 0   
    regularization_loss = 0
    for param in net.parameters():
        regularization_loss += torch.norm(param, p=2)

    total_loss = criterion(output, label) + l2_lambda * regularization_loss
    return total_loss



def save_metrics(metrics, roc_data, prc_data, fold, epoch, data_type):
    os.makedirs(f'results/{fold}/metrics', exist_ok=True)
    os.makedirs(f'results/{fold}/roc', exist_ok=True)
    os.makedirs(f'results/{fold}/prc', exist_ok=True)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(f'results/{fold}/metrics/{data_type}_metrics.csv', index=False)

    roc_df = pd.DataFrame({
        "FPR": roc_data[0],
        "TPR": roc_data[1],
        "AUC": [roc_data[2]] * len(roc_data[0])# AUC is a single value, we repeat it to match the length of FPR and TPR
    })
    roc_df.to_csv(f'results/{fold}/roc/{data_type}_roc_epoch_{epoch + 1}.csv', index=False)

    prc_df = pd.DataFrame({
        "Precision": prc_data[0],
        "Recall": prc_data[1],
        "AUPRC": [prc_data[2]] * len(prc_data[0]),# AUPRC is a single value, we repeat it to match the length of Precision and Recall
        "AP": [prc_data[3]] * len(prc_data[0])
    })
    prc_df.to_csv(f'results/{fold}/prc/{data_type}_prc_epoch_{epoch + 1}.csv', index=False)



class EarlyStopping:
    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_acc = None

    def __call__(self, val_acc, model):
        score = val_acc
        if self.best_score is None:
            self.best_score = score
            # self.save_checkpoint(val_acc, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            # self.save_checkpoint(val_acc, model)
            self.counter = 0

    def save_checkpoint(self, val_acc, model):
        self.best_acc = val_acc
        path = 'best_network.pt'
