import torch
from src.train import do_compute, do_compute_metrics


import numpy as np

def test(test_data_loader, model, device):
    model.eval()
    test_probas_pred = []
    test_ground_truth = []
    count = 0
    with torch.no_grad():
        for i, batch in enumerate(test_data_loader):
            p_score, n_score, probas_pred, ground_truth  = do_compute(batch, device, model)

            test_probas_pred.append(probas_pred)
            test_ground_truth.append(ground_truth)

        test_probas_pred = np.concatenate(test_probas_pred)
        test_ground_truth = np.concatenate(test_ground_truth)

        test_acc, test_auc_roc, test_f1, test_precision,test_recall,test_int_ap, test_ap = do_compute_metrics(test_probas_pred, test_ground_truth)

        print('\n')
        print('============================== Test Result ==============================')
        print(f'\t\ttest_acc: {test_acc:.4f}, test_auc_roc: {test_auc_roc:.4f},test_f1: {test_f1:.4f},test_precision:{test_precision:.4f}')
        print(f'\t\ttest_recall: {test_recall:.4f}, test_int_ap: {test_int_ap:.4f},test_ap: {test_ap:.4f}')





