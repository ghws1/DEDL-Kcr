import numpy as np
import os
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import matplotlib.pyplot as plt
import pandas as pd


def save_predict_result(data, output):
    with open(output, 'w') as f:
        if len(data) > 1:
            for i in range(len(data)):
                f.write('# result for fold %d\n' % (i + 1))
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
        else:
            for i in range(len(data)):
                f.write('# result for predict\n')
                for j in range(len(data[i])):
                    f.write('%d\t%s\n' % (data[i][j][0], data[i][j][2]))
        f.close()
    return None


# Plot the ROC curve and return the AUC value
def plot_roc_curve(data, output, label_column=0, score_column=2):
    datasize = len(data)
    tprs = []
    aucs = []
    fprArray = []
    tprArray = []
    thresholdsArray = []
    mean_fpr = np.linspace(0, 1, 100)
    for i in range(len(data)):
        fpr, tpr, thresholds = roc_curve(data[i][:, label_column], data[i][:, score_column])
        fprArray.append(fpr)
        tprArray.append(tpr)
        thresholdsArray.append(thresholds)
        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
    colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'blueviolet', 'deeppink'])
    fig = plt.figure(figsize=(7, 7), dpi=300)
    for i, color in zip(range(len(fprArray)), colors):
        if datasize > 1:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC fold %d (AUC = %0.3f)' % (i + 1, aucs[i]))
        else:
            plt.plot(fprArray[i], tprArray[i], lw=1, alpha=0.7, color=color,
                     label='ROC (AUC = %0.3f)' % aucs[i])
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',
             label='Random', alpha=.8)
    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    # Calculate the standard deviation
    std_auc = np.std(aucs)
    if datasize > 1:
        plt.plot(mean_fpr, mean_tpr, color='blue',
                 label=r'Mean ROC (AUC = %0.4f $\pm$ %0.3f)' % (mean_auc, std_auc),
                 lw=2, alpha=.9)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    if datasize > 1:
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')
    plt.xlim([0, 1.0])
    plt.ylim([0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.legend(loc="lower right")
    plt.savefig(output, dpi=800, format='svg')
    plt.close(0)
    return mean_auc, aucs


# Calculate and save performance metrics
def calculate_metrics(labels, scores, cutoff=0.5, po_label=1):
    my_metrics = {
        'SN': 'NA',
        'SP': 'NA',
        'ACC': 'NA',
        'MCC': 'NA',
        'Recall': 'NA',
        'Precision': 'NA',
        'F1-score': 'NA',
        'Cutoff': cutoff,
    }

    tp, tn, fp, fn = 0, 0, 0, 0
    for i in range(len(scores)):
        if labels[i] == po_label:
            if scores[i] >= cutoff:
                tp = tp + 1
            else:
                fn = fn + 1
        else:
            if scores[i] < cutoff:
                tn = tn + 1
            else:
                fp = fp + 1

    my_metrics['SN'] = tp / (tp + fn) if (tp + fn) != 0 else 0
    my_metrics['SP'] = tn / (fp + tn) if (fp + tn) != 0 else 0
    my_metrics['ACC'] = (tp + tn) / (tp + fn + tn + fp)
    my_metrics['MCC'] = (tp * tn - fp * fn) / np.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (
                                                                                                                     tp + fp) * (
                                                                                                                     tp + fn) * (
                                                                                                                     tn + fp) * (
                                                                                                                     tn + fn) != 0 else 0
    my_metrics['Precision'] = tp / (tp + fp) if (tp + fp) != 0 else 0
    my_metrics['Recall'] = my_metrics['SN']
    my_metrics['F1-score'] = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    return my_metrics


def calculate_metrics_list(data, label_column=0, score_column=2, cutoff=0.5, po_label=1):
    metrics_list = []
    for i in data:
        metrics_list.append(calculate_metrics(i[:, label_column], i[:, score_column], cutoff=cutoff, po_label=po_label))
    if len(metrics_list) == 1:
        return metrics_list
    else:
        mean_dict = {}
        std_dict = {}
        keys = metrics_list[0].keys()
        for i in keys:
            mean_list = []
            for metric in metrics_list:
                mean_list.append(metric[i])
            mean_dict[i] = np.array(mean_list, dtype=np.float32).sum() / len(metrics_list)
            std_dict[i] = np.array(mean_list, dtype=np.float32).std()
        metrics_list.append(mean_dict)
        metrics_list.append(std_dict)
        return metrics_list


def save_prediction_metrics_list(metrics_list, output):
    if len(metrics_list) == 1:
        with open(output, 'w') as f:
            f.write('Result')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                f.write('value')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()
    else:
        with open(output, 'w') as f:
            f.write('Fold')
            for keys in metrics_list[0]:
                f.write('\t%s' % keys)
            f.write('\n')
            for i in range(len(metrics_list)):
                if i <= len(metrics_list) - 3:
                    f.write('%d' % (i + 1))
                elif i == len(metrics_list) - 2:
                    f.write('mean')
                else:
                    f.write('std')
                for keys in metrics_list[i]:
                    f.write('\t%s' % metrics_list[i][keys])
                f.write('\n')
            f.close()
    return None


def save_val_result(cv_res, outPath, codename):
    out = os.path.join(outPath, codename.lower())
    save_predict_result(cv_res, out + '_pre_cv.txt')
    plot_roc_curve(cv_res, out + '_roc_cv.svg', label_column=0, score_column=2)
    cv_metrics = calculate_metrics_list(cv_res, label_column=0, score_column=2, cutoff=0.5, po_label=1)
    save_prediction_metrics_list(cv_metrics, out + '_metrics_cv.txt')
    return cv_metrics


def mkdir(path):
    path = path.strip()
    path = path.rstrip("\\")
    isExists = os.path.exists(path)
    if not isExists:
        os.makedirs(path)
    else:
        pass


def load_dataset(filepath):
    df = pd.read_csv(filepath, sep=',', header=0)
    seqs = list(df['Sequence'])
    labels = list(df['Label'])
    return seqs, np.array(labels).astype(np.float32)