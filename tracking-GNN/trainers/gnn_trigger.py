"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch
import numpy as np
import sklearn.metrics as metrics
import wandb
# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm, get_grad_norm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        preds = []
        labels = []
        # sum_correct = 0
        # sum_total = 0
        # Loop over training batches
        for i, batch in enumerate(data_loader):
            # batch.w = batch.w[0]
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            self.logger.debug(f'output size: {batch_output.shape}')
            # batch_pred = torch.sigmoid(batch_output)

            # logging.debug(f'match type and y type {type(batch_pred)} {type(batch.y)}')
            # matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            # sum_correct += matches.sum().item()
            # sum_total += matches.numel()

            # batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.y.float().float(), weight=batch.w.float())
            # logging.debug(f'batch w size : {batch.w.shape}')
            # batch_loss = self.loss_func(batch_output, batch.y.float(), weight=batch.w)
            # self.logger.info(f'prediciton: {batch_output}, ground truth: {batch.y}')
            batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.trigger.float())
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()
            preds.append((batch_output>0).cpu().data.numpy())
            trigger = batch.trigger.cpu().numpy()
            labels.append(trigger)
            # predict_noise = batch_pred > 0.5
            # predict_hits = batch_pred < 0.5
            # true_noise = batch.y == 1
            # true_hits = batch.y == 0
            # self.logger.debug(f'\n--train batch predict noise: {predict_noise.sum().item()} true hits: {predict_hits.sum().item()} \n--train batch ground  noise: {true_noise.sum().item()} true hits: {true_hits.sum().item()}')

            # Dump additional debugging information
            if self.logger.isEnabledFor(logging.DEBUG):
                l1 = get_weight_norm(self.model, 1)
                l2 = get_weight_norm(self.model, 2)
                grad_norm = get_grad_norm(self.model)
                self.logger.debug('  train batch %i loss %.4f l1 %.2f l2 %.4f grad %.3f idx %i',
                                  i, batch_loss.item(), l1, l2, grad_norm, batch.i[0].item())

        # Summarize the epoch
        n_batches = i + 1
        summary['lr'] = self.optimizer.param_groups[0]['lr']
        summary['train_loss'] = sum_loss / n_batches
        # summary['train_acc'] = sum_correct / sum_total
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        labels = np.hstack(labels)
        preds = np.hstack(preds)
        result = {'prec': metrics.precision_score(labels, preds, average='macro'),
            'recall': metrics.recall_score(labels, preds, average='macro'),
            'acc': metrics.accuracy_score(labels, preds),
            'F1': metrics.f1_score(labels, preds, average="micro")}
        summary['train_label_acc'] = metrics.accuracy_score(labels, preds)
        summary['train_auroc'] = metrics.roc_auc_score(labels, preds)

        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        # self.logger.info('  Training loss: %.3f acc: %.3f', summary['train_loss'], summary['train_acc'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        if self.use_wandb:
            wandb.log({" train_loss " : summary['train_loss']})
            wandb.log({" train Trigger Accuracy" : summary['train_label_acc']})
            wandb.log({" train Trigger Precion " : result['prec']})
            wandb.log({" train Trigger Recall " : result['recall']})
            wandb.log({" train Trigger F Score for " : result['F1']})
            wandb.log({" train Trigger Roc_auc for " : summary['train_auroc']})
        return summary
    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        if not hasattr(self, 'best_validation_acc'):
            self.best_validation_acc = 0

        MODE = "trigger"
        
        if MODE == "trigger":
            # Prepare summary information
            threshold_list = np.linspace(0, 10, num=11)
            summary = dict()
            tp_list = np.zeros(len(threshold_list))
            fp_list = np.zeros(len(threshold_list))
            fn_list = np.zeros(len(threshold_list))
            tn_list = np.zeros(len(threshold_list))
            preds = []
            labels = []

            # Loop over batches
            for i, batch in enumerate(data_loader):
                batch = batch.to(self.device)
                # Make predictions on this batch
                batch_output = self.model(batch)
                batch_loss = self.loss_func(torch.sigmoid(batch_output), batch.trigger.float())
                preds.append((batch_output>0).cpu().data.numpy())
                trigger = batch.trigger.cpu().numpy()
                labels.append(trigger)
                # Count number of correct predictions
                batch_pred = torch.sigmoid(batch_output).cpu().numpy()
                for i in range(len(threshold_list)):
                    tp_list[i] += np.logical_and(batch_pred >= threshold_list[i], trigger == 1).sum().item()
                    fp_list[i] += np.logical_and(batch_pred >= threshold_list[i], trigger == 0).sum().item()
                    fn_list[i] += np.logical_and(batch_pred < threshold_list[i], trigger == 1).sum().item()
                    tn_list[i] += np.logical_and(batch_pred < threshold_list[i], trigger ==0).sum().item()

            # Summarize the validation epoch
            summary['tp'] = tp_list
            summary['fp'] = fp_list
            summary['fn'] = fn_list
            summary['tn'] = tn_list
            summary['valid_loss'] = batch_loss

            labels = np.hstack(labels)
            preds = np.hstack(preds)
            result = {'prec': metrics.precision_score(labels, preds, average='macro'),
                'recall': metrics.recall_score(labels, preds, average='macro'),
                'acc': metrics.accuracy_score(labels, preds),
                'F1': metrics.f1_score(labels, preds, average="micro")}
            summary['valid_label_acc'] = metrics.accuracy_score(labels, preds)
            summary['valid_auroc'] = metrics.roc_auc_score(labels, preds)

            self.best_validation_acc = max(self.best_validation_acc, summary['valid_label_acc'])
            self.logger.info('Validation loss: %.3f', summary['valid_loss'])
            self.logger.info(f'event classification result: {result}')

            if self.use_wandb:
                wandb.log({" valid_loss " : summary['valid_loss']})
                wandb.log({" valid Trigger Accuracy" : summary['valid_label_acc']})
                wandb.log({"valid Best Trigger Accuracy" : self.best_validation_acc})
                wandb.log({" valid Trigger Precion " : result['prec']})
                wandb.log({" valid Trigger Recall " : result['recall']})
                wandb.log({" valid Trigger F Score " : result['F1']})
                wandb.log({" valid Trigger Roc_auc " : summary['valid_auroc']})

        else:
            # Prepare summary information
            summary = dict()
            sum_loss = 0
            norm_ip = np.load('normal_ip.npz')
            norm_factor = torch.from_numpy(norm_ip['ip'])

            # Loop over batches
            for i, batch in enumerate(data_loader):
                batch.y = batch.y.view(-1,3)
                
                batch = batch.to(self.device)
                norm_factor = norm_factor.to(self.device)
                # Make predictions on this batch
                batch_output = self.model(batch)

                #np.savez("./prediction/"+str(i),**dict(x = batch.x.cpu().numpy(), edge_index = batch.edge_index.cpu().numpy(), ip_gt = (batch.y*norm_factor).cpu().numpy(), ip_pred = (batch_output*norm_factor).cpu().numpy()))
                
                # Count number of correct predictions
                batch_loss = self.loss_func(batch_output*norm_factor, batch.y.float()*norm_factor).item()
                sum_loss += batch_loss
                batch_loss_unnorm = self.loss_func(batch_output, batch.y.float()).item()
                #self.logger.info(f'norm batch loss: {batch_loss},unnorm: {batch_loss_unnorm}')
            
            # Summarize the validation epoch
            n_batches = i + 1
            summary['valid_loss'] = sum_loss / n_batches
            self.logger.info('Validation loss: %.3f', summary['valid_loss'])

        return summary

    @torch.no_grad()
    def predict(self, data_loader):
        preds, targets = [], []
        for batch in data_loader:
            preds.append(torch.sigmoid(self.model(batch)).squeeze(0))
            targets.append(batch.y.squeeze(0))
        return preds, targets

def _test():
    t = SparseGNNTrainer(output_dir='./')
    t.build_model()
