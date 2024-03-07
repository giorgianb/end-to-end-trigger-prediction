"""
This module defines a generic trainer for simple models and datasets.
"""

# System
import logging

# Externals
import torch
import wandb

# Locals
from .gnn_base import GNNBaseTrainer
from utils.checks import get_weight_norm, get_grad_norm
import tqdm

class SparseGNNTrainer(GNNBaseTrainer):
    """Trainer code for sparse GNN."""

    def train_epoch(self, data_loader):
        """Train for one epoch"""
        self.model.train()

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        sum_edge_true = 0
        sum_edge = 0
        sum_nonedge_true = 0
        sum_nonedge = 0
        sigmoid = torch.nn.Sigmoid()

        # Loop over training batches
        for i, batch in enumerate(tqdm.tqdm(data_loader, smoothing=0.0)):
            batch = batch.to(self.device)
            self.model.zero_grad()
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y.float(), weight=batch.w)
            batch_loss.backward()
            self.optimizer.step()
            sum_loss += batch_loss.item()

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            edge_true = ((batch_pred > 0.5) & (batch.y > 0.5)).sum()
            edge_count = (batch.y > 0.5).sum()
            nonedge_true = ((batch_pred < 0.5) & (batch.y < 0.5)).sum()
            nonedge_count = (batch.y < 0.5).sum()
            sum_correct += matches.sum().item()
            sum_edge_true += edge_true.item()
            sum_edge += edge_count.item()
            sum_nonedge_true += nonedge_true.item()
            sum_nonedge += nonedge_count.item()
            sum_total += matches.numel()


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
        summary['train_acc'] = sum_correct / sum_total
        summary['recall'] = sum_edge_true / sum_edge
        summary['precision'] = sum_edge_true / (sum_edge_true + sum_nonedge - sum_nonedge_true)
        summary['f1'] = 2 * summary['recall'] * summary['precision'] / (summary['recall'] + summary['precision'])
        summary['l1'] = get_weight_norm(self.model, 1)
        summary['l2'] = get_weight_norm(self.model, 2)
        summary['grad_norm'] = get_grad_norm(self.model)
        self.logger.debug(' Processed %i batches', n_batches)
        self.logger.debug(' Model LR %f l1 %.2f l2 %.2f',
                          summary['lr'], summary['l1'], summary['l2'])
        self.logger.info('  Training loss: %.3f', summary['train_loss'])
        self.logger.info('  Precision: %.3f', summary['precision'])
        self.logger.info('  Recall: %.3f', summary['recall'])
        self.logger.info('  F1: %.3f', summary['f1'])

        if self.use_wandb:
            phase = 'train'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['train_loss']})
            wandb.log({phase.capitalize() + " Accuracy for tracking" : summary['train_acc']})
            wandb.log({phase.capitalize() + " L1 of parameters for tracking" : summary['l1']})
            wandb.log({phase.capitalize() + " L2 of parameters for tracking" : summary['l2']})
            wandb.log({phase.capitalize() + " Learning rate for tracking " : summary['lr']})
            wandb.log({phase.capitalize() + " Grad Norm for tracking " : summary['grad_norm']})
            wandb.log({phase.capitalize() + " Recall for tracking " : summary['recall']})
            wandb.log({phase.capitalize() + " Precision for tracking " : summary['precision']})
            wandb.log({phase.capitalize() + " F1 for tracking " : summary['f1']})

        return summary

    @torch.no_grad()
    def evaluate(self, data_loader):
        """"Evaluate the model"""
        self.model.eval()
        if not hasattr(self, 'best_validation_f1'):
            self.best_validation_f1 = 0

        # Prepare summary information
        summary = dict()
        sum_loss = 0
        sum_correct = 0
        sum_total = 0
        sum_edge_true = 0
        sum_edge = 0
        sum_nonedge_true = 0
        sum_nonedge = 0
        sigmoid = torch.nn.Sigmoid()

        # Loop over batches
        for i, batch in enumerate(data_loader):
            batch = batch.to(self.device)

            # Make predictions on this batch
            batch_output = self.model(batch)
            batch_loss = self.loss_func(batch_output, batch.y.float()).item()
            sum_loss += batch_loss

            # Count number of correct predictions
            batch_pred = torch.sigmoid(batch_output)
            matches = ((batch_pred > 0.5) == (batch.y > 0.5))
            edge_true = ((batch_pred > 0.5) & (batch.y > 0.5)).sum()
            edge_count = (batch.y > 0.5).sum()
            nonedge_true = ((batch_pred < 0.5) & (batch.y < 0.5)).sum()
            nonedge_count = (batch.y < 0.5).sum()
            sum_correct += matches.sum().item()
            sum_edge_true += edge_true.item()
            sum_edge += edge_count.item()
            sum_nonedge_true += nonedge_true.item()
            sum_nonedge += nonedge_count.item()
            sum_total += matches.numel()
            self.logger.debug(' valid batch %i, loss %.4f', i, batch_loss)

        # Summarize the validation epoch
        n_batches = i + 1
        summary['valid_loss'] = sum_loss / n_batches
        summary['valid_acc'] = sum_correct / sum_total
        summary['recall'] = sum_edge_true / sum_edge
        summary['precision'] = sum_edge_true / (sum_edge_true + sum_nonedge - sum_nonedge_true)
        print(f'{sum_edge_true=} {sum_edge=} {sum_nonedge_true=} {sum_nonedge=}')
        f1 = 2 * summary['recall'] * summary['precision'] / (summary['recall'] + summary['precision']) if (summary['recall'] + summary['precision'] != 0) else 0
        summary['f1'] = f1
        self.logger.debug(' Processed %i samples in %i batches',
                          len(data_loader.sampler), n_batches)
        self.logger.info('  Validation loss: %.3f acc: %.3f edge_true: %.3f edge: %.3f edge_acc: %.3f nonedge_true: %.3f nonedge: %.3f nonedge_acc: %.3f' %
                         (summary['valid_loss'], summary['valid_acc'], sum_edge_true, sum_edge, float(sum_edge_true)/float(sum_edge), sum_nonedge_true, sum_nonedge, float(sum_nonedge_true)/float(sum_nonedge)))
        self.logger.info(' Precision: %.3f', summary['precision'])
        self.logger.info(' Recall: %.3f', summary['recall'])
        self.logger.info(' F1: %.3f', summary['f1'])

        self.best_validation_f1 = max(self.best_validation_f1, summary['f1'])

        
        if self.use_wandb:
            phase = 'valid'
            wandb.log({phase.capitalize() + " Loss for tracking" : summary['valid_loss']})
            wandb.log({phase.capitalize() + " Accuracy for tracking" : summary['valid_acc']})
            wandb.log({phase.capitalize() + " Recall for tracking" : summary['recall']})
            wandb.log({phase.capitalize() + " Precision for tracking" : summary['precision']})
            wandb.log({phase.capitalize() + " F1 for tracking" : summary['f1']})
            wandb.log({phase.capitalize() + " Best F1 for tracking" : self.best_validation_f1})

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
