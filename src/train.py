import matplotlib.pyplot as plt
import time
import numpy as np
import torch
from datetime import datetime
import json
from src.utils.metrics import do_compute_metrics

# Training loop
train_acc_values = []
val_acc_values = []
train_loss_values = []
val_loss_values = []
train_roc_values = []
val_roc_values = []
train_auprc_values = []
val_auprc_values = []
train_f1_values = []
val_f1_values = []
train_recall_values = []
val_recall_values = []
train_precision_values = []
val_precision_values = []


def do_compute(batch, device, model):

        probas_pred, ground_truth = [], []
        pos_tri, neg_tri = batch

        pos_tri = [tensor.to(device=device) for tensor in pos_tri]
        p_score = model(pos_tri)[0]  # Tuple'dan sadece skoru alıyoruz
        probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
        ground_truth.append(np.ones(len(p_score)))

        neg_tri = [tensor.to(device=device) for tensor in neg_tri]
        n_score = model(neg_tri)[0]  # Tuple'dan sadece skoru alıyoruz
        probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
        ground_truth.append(np.zeros(len(n_score)))

        probas_pred = np.concatenate(probas_pred)
        ground_truth = np.concatenate(ground_truth)

        return p_score, n_score, probas_pred, ground_truth


def do_compute_with_attentions(batch, device, model):
    probas_pred, ground_truth = [], []
    pos_tri, neg_tri = batch
    pos_tri = [tensor.to(device=device) for tensor in pos_tri]

    p_score, pos_attentions = model(pos_tri)

    probas_pred.append(torch.sigmoid(p_score.detach()).cpu())
    ground_truth.append(np.ones(len(p_score)))

    neg_tri = [tensor.to(device=device) for tensor in neg_tri]
    n_score, neg_attentions = model(neg_tri)

    probas_pred.append(torch.sigmoid(n_score.detach()).cpu())
    ground_truth.append(np.zeros(len(n_score)))
    probas_pred = np.concatenate(probas_pred)
    ground_truth = np.concatenate(ground_truth)

    attentions = torch.cat([pos_attentions, neg_attentions], dim=0).cpu()

    return p_score, n_score, probas_pred, ground_truth, attentions




def train(model, train_data_loader, val_data_loader, loss_fn,  optimizer, n_epochs, device, scheduler=None):
    print('Starting training at', datetime.today())
    max_acc = 0
    pkl_name = 'modelDDIMI.pkl'
    metrics_file = 'training_metrics.json'

    metrics = {
        'train_acc': [], 'val_acc': [],
        'train_loss': [], 'val_loss': [],
        'train_roc': [], 'val_roc': [],
        'train_auprc': [], 'val_auprc': [],
        'train_f1': [], 'val_f1': [],
        'train_recall': [], 'val_recall': [],
        'train_precision': [], 'val_precision': []
    }

    print('Starting training at', datetime.today())

    for i in range(1, n_epochs+1):
        start = time.time()
        train_loss = 0
        train_loss_pos = 0
        train_loss_neg = 0
        val_loss = 0
        val_loss_pos = 0
        val_loss_neg = 0
        train_probas_pred = []
        train_ground_truth = []
        val_probas_pred = []
        val_ground_truth = []

        for batch in train_data_loader:
            model.train()
            p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
            train_probas_pred.append(probas_pred)
            train_ground_truth.append(ground_truth)
            loss, loss_p, loss_n = loss_fn(p_score, n_score)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(p_score)
        train_loss /= len(train_data_loader.dataset)

        with torch.no_grad():
            train_probas_pred = np.concatenate(train_probas_pred)
            train_ground_truth = np.concatenate(train_ground_truth)

            train_acc, train_auc_roc, train_f1, train_precision,train_recall,train_int_ap, train_ap = do_compute_metrics(train_probas_pred, train_ground_truth)

            for batch in val_data_loader:
                model.eval()
                p_score, n_score, probas_pred, ground_truth = do_compute(batch, device, model)
                val_probas_pred.append(probas_pred)
                val_ground_truth.append(ground_truth)
                loss, loss_p, loss_n = loss_fn(p_score, n_score)
                val_loss += loss.item() * len(p_score)

            val_loss /= len(val_data_loader.dataset)
            val_probas_pred = np.concatenate(val_probas_pred)
            val_ground_truth = np.concatenate(val_ground_truth)
            val_acc, val_auc_roc, val_f1, val_precision,val_recall,val_int_ap, val_ap = do_compute_metrics(val_probas_pred, val_ground_truth)

                    
            metrics['train_acc'].append(float(train_acc))
            metrics['val_acc'].append(float(val_acc))
            metrics['train_loss'].append(float(train_loss))
            metrics['val_loss'].append(float(val_loss))
            metrics['train_roc'].append(float(train_auc_roc))
            metrics['val_roc'].append(float(val_auc_roc))
            metrics['train_auprc'].append(float(train_ap))
            metrics['val_auprc'].append(float(val_ap))
            metrics['train_f1'].append(float(train_f1))
            metrics['val_f1'].append(float(val_f1))
            metrics['train_recall'].append(float(train_recall))
            metrics['val_recall'].append(float(val_recall))
            metrics['train_precision'].append(float(train_precision))
            metrics['val_precision'].append(float(val_precision))

            if val_acc>max_acc:
              max_acc = val_acc
              torch.save(model.state_dict(), pkl_name)
              with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=4)
              

        if scheduler:
            print('scheduler step')
            scheduler.step()

        train_acc_values.append(train_acc)
        val_acc_values.append(val_acc)
        train_loss_values.append(train_loss)
        val_loss_values.append(val_loss)

        train_roc_values.append(train_auc_roc)
        val_roc_values.append(val_auc_roc)

        train_f1_values.append(train_f1)
        val_f1_values.append(val_f1)

        train_recall_values.append(train_recall)
        val_recall_values.append(val_recall)

        train_precision_values.append(train_precision)
        val_precision_values.append(val_precision)
        train_auprc_values.append(train_ap)
        val_auprc_values.append(val_ap)
        

        print(f'Epoch: {i} ({time.time() - start:.4f}s), train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f},'
        f' train_acc: {train_acc:.4f}, val_acc:{val_acc:.4f}')
        print(f'\t\ttrain_roc: {train_auc_roc:.4f}, val_roc: {val_auc_roc:.4f}, train_precision: {train_precision:.4f}, val_precision: {val_precision:.4f}')


