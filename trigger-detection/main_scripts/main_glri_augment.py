import random
import os
import sys
import argparse
import copy
import shutil
import json
import logging
import yaml
import pickle
from pprint import pprint
from datetime import datetime
from functools import partial
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from icecream import ic
from collections import defaultdict
from itertools import chain

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import pickle
import torch_geometric

import tqdm

import wandb

# Change working directory to project's main directory, and add it to path - for library and config usages
project_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
sys.path.append(project_dir)
os.chdir(project_dir)

# Project dependencies
from dataloaders import get_data_loaders
from utils.log import write_checkpoint, load_config, load_checkpoint, config_logging, save_config, print_model_summary, get_terminal_columns, center_text, make_table, numeric_runtime
from utils.losses import SupCon

class ArgDict:
    pass

DEVICE = 'cuda:0'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/glri_augment.yaml')
    argparser.add_argument('-g', '--gpu', default='0', help='The gpu to run on')
    argparser.add_argument('--auto', action='store_true')
    argparser.add_argument('--save', dest='save', action='store_true', help='Whether to save all to disk')
    argparser.add_argument('--no_save', dest='save', action='store_false')
    argparser.set_defaults(save=True, debug_load=False)
    argparser.add_argument('-v', '--verbose', action='store_true')
    argparser.add_argument('--show-config', action='store_true')
    argparser.add_argument('--resume', action='store_true', default=0, help='Resume from last checkpoint')
    
    # Logging
    argparser.add_argument('--name', type=str, default=None,
            help="Run name")
    argparser.add_argument('--use_wandb', action='store_true',
                        help="use wandb project name")
    argparser.add_argument('--skip_wandb_init', action='store_true',
                        help="Skip wandb initialization (helpful if wandb was pre-initialized)")
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

    # Early Stopping
    argparser.add_argument('--early_stopping', action='store_true')
    argparser.set_defaults(early_stopping=False)
    argparser.add_argument('--early_stopping_accuracy', type=float, default=0.65)
    argparser.add_argument('--early_stopping_epoch', type=int, default=1)

    args = argparser.parse_args()

    return args

def calc_metrics(trig, pred, accum_info):
    with torch.no_grad():
        assert len(pred.shape) == 2
        pred = torch.softmax(pred, dim=-1)
        tp = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 1)).item()
        tn = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 0)).item()
        fp = torch.sum((trig == 0) * (torch.argmax(pred, dim=-1) == 1)).item()
        fn = torch.sum((trig == 1) * (torch.argmax(pred, dim=-1) == 0)).item()

        accum_info['true_positives'] += tp
        accum_info['true_negatives'] += tn
        accum_info['false_positives'] += fp
        accum_info['false_negatives'] += fn

    return accum_info


def train(data, model, loss_params, optimizer, epoch, output_dir, finetune_epoch):
    train_info = do_epoch(data, model, loss_params, epoch, finetune_epoch=finetune_epoch, optimizer=optimizer)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, loss_params, epoch, finetune_epoch):
    with torch.no_grad():
        val_info = do_epoch(data, model, loss_params, epoch, optimizer=None, finetune_epoch=finetune_epoch)
    return val_info

def do_epoch(data, model, loss_params, epoch, finetune_epoch=False, optimizer=None):
    if optimizer is None:
        # validation epoch
        model.eval()
    else:
        # train epoch
        model.train()

    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'ri', 
        'auroc', 
        'loss',
        'loss_ce', 
        'loss_kl',
        'loss_mse',
        'fscore', 
        'precision', 
        'recall', 
        'true_positives',
        'true_negatives',
        'false_positives',
        'false_negatives'
    )}

    num_insts = 0
    skipped_batches = 0
    preds = []
    preds_prob = []
    correct = []
    total_size = 0
    total_selected = 0
    sigma, beta, mse_weight = loss_params['sigma'], loss_params['beta'], loss_params['mse_weight']
    for batch in tqdm.tqdm(data):
        tracks = batch.track_vector.to(DEVICE, torch.float)
        n_tracks = batch.n_tracks.to(DEVICE)
        trig = (batch.trigger.to(DEVICE) == 1).long()
        is_trigger_track = batch.is_trigger_track.to(DEVICE, torch.bool)
        batch_size = tracks.shape[0]

        # We do not perturb the hits during the fine-tuning epoch
        if not finetune_epoch and optimizer is not None:
            hits = tracks[..., :15].reshape(*tracks.shape[:-1], 5, 3)
            good_hits = torch.any(hits != 0, dim=-1)
            r = tracks[..., -4] * 100
            valid = r != 0
            centers = tracks[..., -3:-1]
            deltas = hits[..., :2] - centers.unsqueeze(-2)
            angles = torch.atan2(deltas[..., 1], deltas[..., 0])
            dangle = torch.diff(angles, dim=-1)
            ranges = torch.cat([dangle/2, dangle[..., -1].unsqueeze(-1)], dim=-1)
            angle_deltas = 2*ranges*torch.rand(ranges.shape, device=DEVICE) + -ranges
            new_angles = angles + angle_deltas
            hits_deltas = torch.stack([r.unsqueeze(-1)*torch.cos(new_angles), r.unsqueeze(-1)*torch.sin(new_angles)], dim=-1)
            new_hits = torch.cat([centers.unsqueeze(-2) + hits_deltas, hits[..., -1].unsqueeze(-1)], dim=-1)
            tracks[..., :15] = valid.unsqueeze(-1)*(new_hits*good_hits.unsqueeze(-1)).reshape(tracks[..., :15].shape) + (~valid.unsqueeze(-1))*tracks[..., :15]

        # Fit z coordinate to x/y
        #A = torch.cat([torch.ones_like(hits[..., :1]), hits[..., :2]], dim=-1)
        #M = torch.einsum('...ij,...jk', A.transpose(-1, -2), A)
        #b = torch.einsum('...ij,...j', A.transpose(-1, -2), hits[..., -1])
        #m = torch.linalg.solve(M, b)

        mask = torch.zeros((tracks.shape[0], tracks.shape[1]))
        for i, n_track in enumerate(n_tracks):
            mask[i, :n_track] = 1
        mask = mask.to(DEVICE)

        pred_labels, *rest = model(tracks, mask, finetune_epoch=finetune_epoch)
        loss = 0
        ce_loss = F.cross_entropy(pred_labels, trig)
        loss += ce_loss
        accum_info['loss_ce'] += ce_loss.item() * batch_size
        if finetune_epoch:
            pred_sigma, good_hits = rest
            true_sigma = sigma * torch.eye(pred_sigma.shape[-1], device=DEVICE).reshape(1, 1, 1, 3, 3).repeat(
                    pred_sigma.shape[0], #batch
                    pred_sigma.shape[1], #track
                    pred_sigma.shape[2], #hits
                    1,
                    1,
            )

            mask = mask.to(torch.bool)
            loc = torch.zeros(*pred_sigma.shape[:-1]).to(DEVICE)
            pred_normal = torch.distributions.MultivariateNormal(loc, pred_sigma)
            true_normal = torch.distributions.MultivariateNormal(loc, true_sigma)
            kl_loss = torch.distributions.kl.kl_divergence(pred_normal, true_normal)
            kl_loss *= good_hits
            # in the paper, the sum is taken over the vertices
            kl_loss = torch.sum(kl_loss, dim=(-1, -2))
            kl_loss = torch.mean(kl_loss, dim=0)

            loss += beta*kl_loss
            accum_info['loss_kl'] += kl_loss
            pred = pred_labels.max(dim=1)[1]
            preds.extend(pred.cpu().data.numpy())
            preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
            correct.extend(trig.detach().cpu().numpy().flatten())

        else:
            embeddings_perturbed = rest[0]
            pred_perturbed = pred_labels
            tracks = batch.track_vector.to(DEVICE)
            pred_unperturbed, embeddings_unperturbed = model(tracks, mask, finetune_epoch=finetune_epoch)
            ce_loss = F.cross_entropy(pred_unperturbed, trig)
            loss += ce_loss
            accum_info['loss_ce'] += ce_loss.item() * batch_size
            loss_mse = F.mse_loss(embeddings_perturbed, embeddings_unperturbed)
            loss += mse_weight*loss_mse
            accum_info['loss_mse'] += loss_mse.item() * batch_size

            pred = pred_perturbed.max(dim=1)[1]
            preds.extend(pred.cpu().data.numpy())
            preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
            correct.extend(trig.detach().cpu().numpy().flatten())

            pred = pred_unperturbed.max(dim=1)[1]
            preds.extend(pred.cpu().data.numpy())
            preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
            correct.extend(trig.detach().cpu().numpy().flatten())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(trig, pred_labels, accum_info)
        accum_info['loss'] += loss.item()
        num_insts += batch_size

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    if num_insts > 0:
        accum_info['loss'] /= num_insts
        accum_info['loss_ce'] /= num_insts
        accum_info['loss_kl'] /= num_insts
        accum_info['ri'] = (tp + tn)/(tp + tn + fp + fn)
        accum_info['precision'] = tp / (tp + fp) if tp + fp != 0 else 0
        accum_info['recall'] = tp / (tp + fn) if tp + fn != 0 else 0
        accum_info['fscore'] = (2 * tp)/(2 * tp + fp + fn) if (2 * tp + fp + fn) != 0 else 0
    correct = np.array(correct)
    preds = np.array(preds)
    preds_prob = np.array(preds_prob)

    try:
        accum_info['auroc'] = roc_auc_score(correct, preds_prob)
    except ValueError:
        accum_info['auroc'] = 0
           
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]

    print('Skipped batches:', skipped_batches)


    return accum_info

def main():
     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    execute_training(args, config)

def execute_training(args, config):
    global DEVICE

    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


    config['output_dir'] = os.path.join(config['output_dir'], f'experiment_{start_time:%Y-%m-%d_%H:%M:%S}')
    os.makedirs(config['output_dir'], exist_ok=True)

    # Setup logging
    file_handler = config_logging(verbose=args.verbose, output_dir=config['output_dir'],
                   append=args.resume, rank=0)

    logging.info('Command line config: %s' % args)
    logging.info('Configuration: %s', config)
    logging.info('Saving job outputs to %s', config['output_dir'])

    # Save configuration in the outptut directory
    save_config(config)
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"  # uncomment only for CUDA error debugging
    # os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    torch.cuda.set_device(int(args.gpu))
    DEVICE = 'cuda:' + str(args.gpu)

    name = config['wandb']['run_name'] + f'-experiment_{start_time:%Y-%m-%d_%H:%M:%S}'
    logging.info(name)

    if args.use_wandb and not args.skip_wandb_init:
        wandb.init(
            project=config['wandb']['project_name'],
            name=name,
            tags=config['wandb']['tags'],
            config=config
        )

    # Load data
    logging.info('Loading training, validation, and test data')
    dconfig = copy.copy(config['data'])

    train_data, val_data, test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g training samples', len(train_data.dataset))
    logging.info('Loaded %g validation samples', len(val_data.dataset))
    logging.info('Loaded %g test samples', len(test_data.dataset))

    mconfig = copy.copy(config['model'])
    from models.Bipartite_Attention_gLRI import Bipartite_Attention as Model

    model = Model(
        **mconfig
    )
    model = model.to(DEVICE)

    # Optimizer
    oconfig = config['optimizer']
    params = [
            {'params': chain(model.interpreter.parameters(), model.noiser.parameters()), 'lr': oconfig['learning_rate']},
            {'params': model.classifier.parameters(), 'lr' : oconfig['learning_rate']}
    ]
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=params,
                lr=oconfig['learning_rate'], 
                weight_decay=oconfig['weight_decay'], 
                betas=[oconfig['beta_1'], oconfig['beta_2']],
                eps=oconfig['eps']
        )
    elif oconfig['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=params, lr=oconfig['learning_rate'], momentum=oconfig['momentum'], weight_decay=oconfig['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]["type"]} not implemented.')


    decay_rate = oconfig["learning_rate_decay_rate"]
    def lr_schedule_noiser(epoch):
        return decay_rate**epoch

    def lr_schedule_classifier(epoch):
        return decay_rate**epoch

    def lr_schedule_finetune(epoch):
        return oconfig['finetune_factor']*(decay_rate**(epoch + config['epochs']))

    lr_scheduler_classifier = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule_noiser, lr_schedule_classifier])
    lr_scheduler_finetune = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_schedule_noiser, lr_schedule_finetune])
    lr_scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, [lr_scheduler_classifier, lr_scheduler_finetune], milestones=[config['epochs']])

    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of model parameters is {num_params}')

    # Metrics
    train_loss = np.empty(config['epochs'] + config['finetune_epochs'], float)
    train_ri = np.empty(config['epochs'] + config['finetune_epochs'], float)
    val_loss = np.empty(config['epochs'] + config['finetune_epochs'], float)
    val_ri = np.empty(config['epochs'] + config['finetune_epochs'], float)

    best_epoch = -1
    best_val_ri = 0
    best_val_auroc = 0
    best_classifier_val_ri = 0
    best_model = None
    for epoch in range(1, config['epochs'] + config['finetune_epochs'] + 1):
        finetune_epoch = epoch > config['epochs']
        train_info = train(train_data, model, config['loss'], optimizer, epoch, config['output_dir'], finetune_epoch=finetune_epoch)
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
            ('CE loss', f"{train_info['loss_ce']:.6f}"),
            ('KL loss', f"{train_info['loss_kl']:.6f}"),
            ('Rand Index', f"{train_info['ri']:.6f}"),
            ('F-score', f"{train_info['fscore']:.4f}"),
            ('Recall', f"{train_info['recall']:.4f}"),
            ('Precision', f"{train_info['precision']:.4f}"),
            ('True Positives', f"{train_info['true_positives']}"),
            ('False Positives', f"{train_info['false_positives']}"),
            ('True Negatives', f"{train_info['true_negatives']}"),
            ('False Negatives', f"{train_info['false_negatives']}"),
            ('AUC Score', f"{train_info['auroc']:.6f}"),
            ('Runtime', f"{train_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Training - {epoch:4}", ' '),
            table
        )))

        train_loss[epoch-1], train_ri[epoch-1] = train_info['loss'], train_info['ri']
        if args.use_wandb:
            wandb.log({"Train Loss" : train_info['loss']}, step=epoch)
            wandb.log({"Train CE Loss" : train_info['loss_ce']}, step=epoch)
            wandb.log({"Train KL Loss" : train_info['loss_kl']}, step=epoch)
            wandb.log({"Train Accuracy" : train_info['ri']}, step=epoch)
            wandb.log({"Train Precision" : train_info['precision']}, step=epoch)
            wandb.log({"Train Recall": train_info['recall']}, step=epoch)
            wandb.log({"Train F-score": train_info['fscore']}, step=epoch)
            wandb.log({"Train AUROC": train_info['auroc']}, step=epoch)
            wandb.log({"Train Run-Time": numeric_runtime(train_info['run_time'])}, step=epoch)

        val_info = evaluate(val_data, model, config['loss'], epoch, finetune_epoch=finetune_epoch)
        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
            ('CE loss', f"{val_info['loss_ce']:.6f}"),
            ('KL loss', f"{val_info['loss_kl']:.6f}"),
            ('Rand Index', f"{val_info['ri']:.6f}"),
            ('F-score', f"{val_info['fscore']:.4f}"),
            ('Recall', f"{val_info['recall']:.4f}"),
            ('Precision', f"{val_info['precision']:.4f}"),
            ('True Positives', f"{val_info['true_positives']}"),
            ('False Positives', f"{val_info['false_positives']}"),
            ('True Negatives', f"{val_info['true_negatives']}"),
            ('False Negatives', f"{val_info['false_negatives']}"),
            ('AUC Score', f"{val_info['auroc']:.6f}"),
            ('Runtime', f"{val_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            center_text(f"Validation - {epoch:4}", ' '),
            table
            )))

        if val_info['ri'] > best_classifier_val_ri:
            best_classifier_val_ri = val_info['ri']
            best_model = copy.deepcopy(model)

        if val_info['ri'] > best_val_ri and finetune_epoch:
            best_val_ri = val_info['ri']
            best_val_auroc = val_info['auroc']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']
        if args.use_wandb:
            wandb.log({"Validation Loss" : val_info['loss']}, step=epoch)
            wandb.log({"Validation CE Loss" : val_info['loss_ce']}, step=epoch)
            wandb.log({"Validation KL Loss" : val_info['loss_kl']}, step=epoch)
            wandb.log({"Validation Accuracy" : val_info['ri']}, step=epoch)
            wandb.log({"Validation Precision" : val_info['precision']}, step=epoch)
            wandb.log({"Validation Recall": val_info['recall']}, step=epoch)
            wandb.log({"Validation F-Score": val_info['fscore']}, step=epoch)
            wandb.log({"Validation AUROC": val_info['auroc']}, step=epoch)
            wandb.log({"Validation Run-Time": numeric_runtime(val_info['run_time'])}, step=epoch)
            wandb.log({"Best Classifier Validation Accuracy": best_classifier_val_ri}, step=epoch)
            wandb.log({"Best Validation Accuracy": best_val_ri}, step=epoch)


        if args.early_stopping and epoch >= args.early_stopping_epoch and best_classifier_val_ri < args.early_stopping_accuracy:
            break

        lr_scheduler.step()
        
    
    del train_data, val_data


    logging.info(f'Best validation accuracy: {best_val_ri:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')

    test_info = evaluate(test_data, best_model, config['loss'], config['epochs'] + 1, finetune_epoch=True)
    table = make_table(
        ('Total loss', f"{test_info['loss']:.6f}"),
        ('CE loss', f"{test_info['loss_ce']:.6f}"),
        ('KL loss', f"{test_info['loss_kl']:.6f}"),
        ('Rand Index', f"{test_info['ri']:.6f}"),
        ('F-score', f"{test_info['fscore']:.4f}"),
        ('Recall', f"{test_info['recall']:.4f}"),
        ('Precision', f"{test_info['precision']:.4f}"),
        ('True Positives', f"{test_info['true_positives']}"),
        ('False Positives', f"{test_info['false_positives']}"),
        ('True Negatives', f"{test_info['true_negatives']}"),
        ('False Negatives', f"{test_info['false_negatives']}"),
        ('AUC Score', f"{test_info['auroc']:.6f}"),
        ('Runtime', f"{test_info['run_time']}")
    )
    logging.info('\n'.join((
        '',
        center_text(f"Test", ' '),
        table
        )))


    if args.use_wandb:
        wandb.log({"Test Loss" : test_info['loss']}, step=config['epochs'] + 1)
        wandb.log({"Test CE Loss" : test_info['loss_ce']})
        wandb.log({"Test KL Loss" : test_info['loss_kl']})
        wandb.log({"Test Accuracy" : test_info['ri']})
        wandb.log({"Test Precision" : test_info['precision']})
        wandb.log({"Test Recall": test_info['recall']})
        wandb.log({"Test F-Score": test_info['fscore']})
        wandb.log({"Test AUROC": test_info['auroc']})
        wandb.log({"Test Run-Time": numeric_runtime(test_info['run_time'])})

    # Saving to disk
    if args.save:
        output_dir = os.path.join(config['output_dir'], 'summary')
        i = 0
        while True:
            if not os.path.isdir(output_dir):
                os.makedirs(output_dir)  # raises error if dir already exists
                break
            i += 1
            output_dir = output_dir[:-1] + str(i)
            if i > 9:
                logging.info(f'Cannot save results on disk. (tried to save as {output_dir})')
                return

        logging.info(f'Saving all to {output_dir}')
        torch.save(best_model.state_dict(), os.path.join(output_dir, "exp_model.pt"))
        shutil.copyfile(__file__, os.path.join(output_dir, 'main.py'))
        shutil.copytree('models/', os.path.join(output_dir, 'models/'))
        results_dict = {'train_loss': train_loss,
                        'train_ri': train_ri,
                        'val_loss': val_loss,
                        'val_ri': val_ri}
        df = pd.DataFrame(results_dict)
        df.index.name = 'epochs'
        df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        best_dict = {'best_val_ri': best_val_ri, 'best_epoch': best_epoch}
        best_df = pd.DataFrame(best_dict, index=[0])
        best_df.to_csv(os.path.join(output_dir, "best_val_results.csv"), index=False)

    logging.shutdown()



if __name__ == '__main__':
    main()
