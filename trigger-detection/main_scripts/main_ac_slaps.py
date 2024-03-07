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
from torch.utils.tensorboard import SummaryWriter

class ArgDict:
    pass

DEVICE = 'cuda:0'

def parse_args():
    """
    Define and retrieve command line arguements
    :return: argparser instance
    """
    argparser = argparse.ArgumentParser(description=__doc__)
    argparser.add_argument('--config', default='configs/ac_slaps.yaml')
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
    argparser.add_argument('--log_interval', type=int, default=25,
            help="Number of steps between logging key stats")
    argparser.add_argument('--print_interval', type=int, default=250,
            help="Number of steps between printing key stats")

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


def train(data, model, dae_weight, optimizer, epoch, output_dir):
    train_info = do_epoch(data, model, epoch, dae_weight, optimizer)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def train_contrast(trigger_data, nontrigger_data, model, contrast_loss_fn, optimizer, epoch, output_dir, n_positive, n_negative, drop_probability):
    train_info = do_contrast_epoch(trigger_data, nontrigger_data, model, contrast_loss_fn, epoch, n_positive, n_negative, drop_probability, optimizer)
    write_checkpoint(checkpoint_id=epoch, model=model, optimizer=optimizer, output_dir=output_dir)
    return train_info

def evaluate(data, model, dae_weight, epoch):
    with torch.no_grad():
        val_info = do_epoch(data, model, dae_weight, epoch, optimizer=None)
    return val_info

def evaluate_contrast(trigger_data, nontrigger_data, model, contrast_loss_fn, epoch, n_positive, n_negative, drop_probability):
    with torch.no_grad():
        val_info = do_contrast_epoch(trigger_data, nontrigger_data, model, contrast_loss_fn, epoch, n_positive, n_negative, drop_probability, optimizer=None)
    return val_info


def do_contrast_epoch(trigger_data, nontrigger_data, model, contrast_loss_fn, epoch, n_positive, n_negative, drop_probability, optimizer=None):
    if optimizer is None:
        # validation epoch
        model.eval()
    else:
        # train epoch
        model.train()


    start_time = datetime.now()

    # Iterate over batches
    accum_info = {k: 0.0 for k in (
        'loss',
        'loss_contrast'
    )}

    num_insts = 0
    skipped_batches = 0
    preds = []
    preds_prob = []
    correct = []
    total_size = 0
    total_selected = 0
    
    for trigger_batch, nontrigger_batch in tqdm.tqdm(zip(trigger_data, nontrigger_data), total=len(trigger_data)):
        loss = 0

        trigger_tracks = trigger_batch.track_vector.to(DEVICE, torch.float)
        nontrigger_tracks = nontrigger_batch.track_vector.to(DEVICE, torch.float)

        trigger_n_tracks = trigger_batch.n_tracks.to(DEVICE)
        nontrigger_n_tracks = nontrigger_batch.n_tracks.to(DEVICE)

        batch_size = trigger_tracks.shape[0]
        is_trigger_track = trigger_batch.is_trigger_track.to(DEVICE)

        trigger_mask = torch.zeros((trigger_tracks.shape[0], trigger_tracks.shape[1]), dtype=torch.bool)
        for i, n_track in enumerate(trigger_n_tracks):
            trigger_mask[i, :n_track] = True
        trigger_mask = trigger_mask.to(DEVICE)

        nontrigger_mask = torch.zeros((nontrigger_tracks.shape[0], nontrigger_tracks.shape[1]), dtype=torch.bool)
        for i, n_track in enumerate(nontrigger_n_tracks):
            nontrigger_mask[i, :n_track] = True
        nontrigger_mask = nontrigger_mask.to(DEVICE)

        trigger_query_embeddings = model.generate_embedding(trigger_tracks, trigger_mask.to(torch.long))
        nontrigger_query_embeddings = model.generate_embedding(nontrigger_tracks, nontrigger_mask.to(torch.long))

        # So now we generate positive and negative examples in the following manner:
        # Trigger events:
        # Positive Samples:
        # a) Off-diagonal embeddings
        # b) Embeddings with random background tracks dropped
        # c) Embeddings with only signal tracks
        # Negative Samples:
        # a) Non-Trigger Events
        # b) Embeddings with trigger tracks dropped
        # c) Embeddings with trigger tracks dropped and some random background tracks

        # Add embeddings with random background tracks dropped
        trigger_pos_mask = torch.rand((trigger_mask.shape[0], n_positive, trigger_mask.shape[1])).to(DEVICE) >= drop_probability
        # Add embeddings with only signal tracks
        trigger_pos_mask = torch.cat([trigger_pos_mask, is_trigger_track.unsqueeze(1)], dim=1)


        # Ensure all trigger tracks are included
        trigger_pos_mask = trigger_pos_mask | is_trigger_track.unsqueeze(1)
        # Ensure all tracks are valid
        trigger_pos_mask = trigger_pos_mask & trigger_mask.unsqueeze(1)

        trigger_pos_tracks = trigger_tracks.unsqueeze(0).repeat((n_positive + 1, 1, 1, 1)).transpose(0, 1)
        # (B, N, F)
        trigger_pos_tracks = trigger_pos_tracks.reshape(
            trigger_tracks.shape[0]*(n_positive + 1),
            trigger_tracks.shape[1],
            trigger_tracks.shape[2]
        )
        # (B, N)
        trigger_pos_mask = trigger_pos_mask.reshape(
                trigger_tracks.shape[0]*(n_positive + 1),
                trigger_mask.shape[1]
        ).to(torch.long)

        trigger_pos_embeddings = model.generate_embedding(trigger_pos_tracks, trigger_pos_mask)
        trigger_pos_embeddings = trigger_pos_embeddings.reshape((
            trigger_tracks.shape[0], 
            n_positive + 1, 
            trigger_pos_embeddings.shape[1]
        ))
        
        # Add off-diagonal trigger examples
        batch_size = trigger_query_embeddings.shape[0]
        off_diag_embeddings = trigger_query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        off_diag_embeddings = off_diag_embeddings[~torch.eye(batch_size, dtype=torch.bool).to(DEVICE)]
        off_diag_embeddings = off_diag_embeddings.reshape(batch_size-1, batch_size, -1).transpose(0, 1)
        trigger_pos_embeddings = torch.cat([trigger_pos_embeddings, off_diag_embeddings], dim=1)

        # Embeddings with trigger tracks dropped and random background tracks
        trigger_neg_mask = torch.rand((trigger_mask.shape[0], n_negative, trigger_mask.shape[1])).to(DEVICE) >= drop_probability

        # Add embeddings with only the background tracks
        trigger_neg_mask = torch.cat([trigger_neg_mask, ~is_trigger_track.unsqueeze(1)], dim=1)

        # Ensure that no trigger tracks are included
        trigger_neg_mask = trigger_neg_mask & ~is_trigger_track.unsqueeze(1)
        # Ensure that all tracks are valid
        trigger_neg_mask = trigger_neg_mask & trigger_mask.unsqueeze(1)


        # (B, N, F)
        trigger_neg_tracks = trigger_tracks.unsqueeze(0).repeat((n_negative + 1, 1, 1, 1)).transpose(0, 1)
        trigger_neg_tracks = trigger_neg_tracks.reshape(
            trigger_tracks.shape[0]*(n_negative + 1),
            trigger_tracks.shape[1],
            trigger_tracks.shape[2]
        )
        trigger_neg_mask = trigger_neg_mask.reshape(
                trigger_tracks.shape[0]*(n_negative + 1),
                trigger_mask.shape[1]
        ).to(torch.long)

        trigger_neg_embeddings = model.generate_embedding(trigger_neg_tracks, trigger_neg_mask)
        trigger_neg_embeddings = trigger_neg_embeddings.reshape((
            trigger_tracks.shape[0], 
            n_negative + 1, 
            trigger_neg_embeddings.shape[1]
        ))

        # Add non-trigger embeddings
        batch_size = nontrigger_query_embeddings.shape[0]
        trigger_neg_embeddings = torch.cat([
            trigger_neg_embeddings,
            nontrigger_query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        ], dim=1)

        # Non-Trigger Events:
        # Positive Samples:
        # a) Off-Diagonal Embeddings
        # b) Embeddings with random background tracks dropped
        # Negative Samples:
        # a) Trigger Events
        # b) Trigger Events with random background tracks dropped
        nontrigger_pos_mask = torch.rand((nontrigger_mask.shape[0], n_positive, nontrigger_mask.shape[1])).to(DEVICE) >= drop_probability
        # Ensure all tracks are valid
        nontrigger_pos_mask = nontrigger_pos_mask & nontrigger_mask.unsqueeze(1)

        nontrigger_pos_tracks = nontrigger_tracks.unsqueeze(0).repeat((n_positive, 1, 1, 1)).transpose(0, 1)
        nontrigger_pos_tracks = nontrigger_pos_tracks.reshape(
            nontrigger_tracks.shape[0]*n_positive,
            nontrigger_tracks.shape[1],
            nontrigger_tracks.shape[2]
        )
        nontrigger_pos_mask = nontrigger_pos_mask.reshape(
                nontrigger_tracks.shape[0]*n_positive,
                nontrigger_mask.shape[1]
        ).to(torch.long)

        nontrigger_pos_embeddings = model.generate_embedding(nontrigger_pos_tracks, nontrigger_pos_mask)
        nontrigger_pos_embeddings = nontrigger_pos_embeddings.reshape((
            nontrigger_tracks.shape[0], 
            n_positive,
            nontrigger_pos_embeddings.shape[1]
        ))

        
        # Add off-diagonal nontrigger examples
        batch_size = nontrigger_query_embeddings.shape[0]
        off_diag_embeddings = nontrigger_query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        off_diag_embeddings = off_diag_embeddings[~torch.eye(batch_size, dtype=torch.bool).to(DEVICE)]
        off_diag_embeddings = off_diag_embeddings.reshape(batch_size-1, batch_size, -1).transpose(0, 1)
        nontrigger_pos_embeddings = torch.cat([nontrigger_pos_embeddings, off_diag_embeddings], dim=1)

        nontrigger_neg_embeddings = torch.cat([
            trigger_pos_embeddings,
            trigger_query_embeddings.unsqueeze(0).repeat(batch_size, 1, 1)
        ], dim=1)
        

        contrast_loss = contrast_loss_fn(
                trigger_query_embeddings, 
                trigger_pos_embeddings, 
                trigger_neg_embeddings
                )
        contrast_loss += contrast_loss_fn(
                nontrigger_query_embeddings,
                nontrigger_pos_embeddings,
                nontrigger_neg_embeddings
        )
        loss += contrast_loss
        accum_info['loss_contrast'] += contrast_loss.item() * batch_size


        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info['loss'] += loss.item() * batch_size
        num_insts += batch_size

    if num_insts > 0:
        accum_info['loss'] /= num_insts
        accum_info['loss_contrast'] /= num_insts
    accum_info['run_time'] = datetime.now() - start_time
    accum_info['run_time'] = str(accum_info['run_time']).split(".")[0]


    return accum_info


def do_epoch(data, model, epoch, dae_weight, optimizer=None):
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

    mse_loss = nn.MSELoss()
    for batch in tqdm.tqdm(data):
        tracks = batch.track_vector.to(DEVICE, torch.float)
        n_tracks = batch.n_tracks.to(DEVICE)
        trig = (batch.trigger.to(DEVICE) == 1).long()
        batch_size = tracks.shape[0]

        mask = torch.zeros((tracks.shape[0], tracks.shape[1]))
        for i, n_track in enumerate(n_tracks):
            mask[i, :n_track] = 1
        mask = mask.to(DEVICE)

        pred_labels, node_embeddings, reconstructed = model(tracks, mask)
        loss = 0
        ce_loss = F.cross_entropy(pred_labels, trig)
        loss += ce_loss
        accum_info['loss_ce'] += ce_loss.item() * batch_size
        loss += dae_weight * mse_loss(node_embeddings, reconstructed)

        pred = pred_labels.max(dim=1)[1]
        preds.extend(pred.cpu().data.numpy())
        preds_prob.extend(nn.Softmax(dim=1)(pred_labels)[:, 1].detach().cpu().numpy().flatten())
        correct.extend(trig.detach().cpu().numpy().flatten())

        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        accum_info = calc_metrics(trig, pred_labels, accum_info)
        accum_info['loss'] += loss.item() * batch_size
        num_insts += batch_size

    tp = accum_info['true_positives']
    tn = accum_info['true_negatives']
    fp = accum_info['false_positives']
    fn = accum_info['false_negatives']

    if num_insts > 0:
        accum_info['loss'] /= num_insts
        accum_info['loss_ce'] /= num_insts
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
    global DEVICE

    start_time = datetime.now()
    seed = 42
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

     # Parse the command line
    args = parse_args()

    # Load configuration
    config = load_config(args.config)

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

    if args.use_wandb:
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

    logging.info('Loading contrastive training, validation, and test data')
    dconfig = copy.copy(config['data'])
    dconfig['use_trigger'] = True
    dconfig['use_nontrigger'] = False

    trigger_train_data, trigger_val_data, trigger_test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g trigger contrastive training samples', len(trigger_train_data.dataset))
    logging.info('Loaded %g trigger contrastive validation samples', len(trigger_val_data.dataset))
    logging.info('Loaded %g trigger contrastive test samples', len(trigger_test_data.dataset))

    dconfig = copy.copy(config['data'])
    dconfig['use_nontrigger'] = True
    dconfig['use_trigger'] = False

    nontrigger_train_data, nontrigger_val_data, nontrigger_test_data = get_data_loaders(**dconfig)
    logging.info('Loaded %g nontrigger contrastive training samples', len(nontrigger_train_data.dataset))
    logging.info('Loaded %g nontrigger contrastive validation samples', len(nontrigger_val_data.dataset))
    logging.info('Loaded %g nontrigger contrastive test samples', len(nontrigger_test_data.dataset))



    mconfig = copy.copy(config['model'])
    from models.Bipartite_Attention_SLAPS import Bipartite_Attention as Model

    model = Model(
        **mconfig
    )
    model = model.to(DEVICE)

    # Optimizer
    if config['optimizer']['type'] == 'Adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=config['optimizer']['learning_rate'], weight_decay=config['optimizer']['weight_decay'])
    elif config['optimizer']['type'] == 'SGD':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=config['optimizer']['learning_rate'], momentum=config['optimizer']['momentum'], weight_decay=config['optimizer']['weight_decay'])
    else:
        raise NotImplementedError(f'Optimizer {config["optimizer"]["type"]} not implemented.')


    def lr_schedule(epoch):
        if epoch > 10 and epoch <= 20:
            return 0.1
        elif epoch > 20 and epoch <= 40:
            return 0.01
        elif epoch > 40 and epoch <= 80:
            return 0.001
        elif epoch > 80:
            return 0.0001
        else:
            return 1

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_schedule)

    print_model_summary(model)
    model = model.to(DEVICE)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logging.info(f'The number of model parameters is {num_params}')

    contrast_loss_fn = SupCon(temperature=config['contrast']['loss']['temperature']).to(DEVICE)

    n_positive = config['contrast']['n_positive']
    n_negative = config['contrast']['n_negative']
    drop_probability = config['contrast']['drop_probability']
    epoch = 0

    # Metrics
    train_loss = np.empty(config['epochs'], float)
    train_ri = np.empty(config['epochs'], float)
    val_loss = np.empty(config['epochs'], float)
    val_ri = np.empty(config['epochs'], float)

    best_epoch = -1
    best_val_ri = -1
    best_val_auroc = -1
    best_model = None
    for epoch in range(1, config['epochs'] + 1):
        train_info = train_contrast(trigger_train_data, nontrigger_train_data, model, contrast_loss_fn, optimizer, epoch, config['output_dir'], n_negative=n_negative, n_positive=n_positive, drop_probability=drop_probability)
        table = make_table(
            ('Contrast Loss', f"{train_info['loss_contrast']:.6f}"),
            ('Runtime', f"{train_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            "#" * get_terminal_columns(),
            center_text(f"Contrastive Training - {epoch:4}", ' '),
            table
        )))
        if args.use_wandb:
            wandb.log({"Train Contrastive Loss" : train_info['loss_contrast']})
            wandb.log({"Contrastive Train Run Time": numeric_runtime(train_info['run_time'])})

        val_info = evaluate_contrast(trigger_val_data, nontrigger_val_data, model, contrast_loss_fn, epoch, n_negative=n_negative, n_positive=n_positive, drop_probability=drop_probability)
        table = make_table(
            ('Contrast loss', f"{val_info['loss_contrast']:.6f}"),
            ('Runtime', f"{val_info['run_time']}")
        )

        logging.info('\n'.join((
            '',
            center_text(f"Contrastive Validation - {epoch:4}", ' '),
            table
            )))
        if args.use_wandb:
            wandb.log({"Validation Contrastive Loss" : val_info['loss_contrast']})
            wandb.log({"Contrastive Validation Run Time": numeric_runtime(val_info['run_time'])})

        train_info = train(train_data, model, config['slaps']['dae_weight'], optimizer, epoch, config['output_dir'])
        table = make_table(
            ('Total loss', f"{train_info['loss']:.6f}"),
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
            wandb.log({"Train Loss" : train_info['loss']})
            wandb.log({"Train Accuracy" : train_info['ri']})
            wandb.log({"Train Precision" : train_info['precision']})
            wandb.log({"Train Recall": train_info['recall']})
            wandb.log({"Train F-score": train_info['fscore']})
            wandb.log({"Train AUROC": train_info['auroc']})
            wandb.log({"Train Run-Time": numeric_runtime(train_info['run_time'])})

        val_info = evaluate(val_data, model, config['slaps']['dae_weight'], epoch)
        table = make_table(
            ('Total loss', f"{val_info['loss']:.6f}"),
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

        val_loss[epoch-1], val_ri[epoch-1] = val_info['loss'], val_info['ri']
        if args.use_wandb:
            wandb.log({"Validation Loss" : val_info['loss']})
            wandb.log({"Validation Accuracy" : val_info['ri']})
            wandb.log({"Validation Precision" : val_info['precision']})
            wandb.log({"Validation Recall": val_info['recall']})
            wandb.log({"Validation F-Score": val_info['fscore']})
            wandb.log({"Validation AUROC": val_info['auroc']})
            wandb.log({"Validation Run-Time": numeric_runtime(val_info['run_time'])})

        if val_info['ri'] > best_val_ri:
            best_val_ri = val_info['ri']
            best_val_auroc = val_info['auroc']
            best_epoch = epoch
            best_model = copy.deepcopy(model)

        lr_scheduler.step()
        
    
    del train_data, val_data


    logging.info(f'Best validation accuracy: {best_val_ri:.4f}, best epoch: {best_epoch}.')
    logging.info(f'Training runtime: {str(datetime.now() - start_time).split(".")[0]}')
    test_info = evaluate_contrast(trigger_test_data, nontrigger_test_data, best_model, contrast_loss_fn, epoch, n_negative=n_negative, n_positive=n_positive, drop_probability=drop_probability)
    table = make_table(
        ('Contrast loss', f"{test_info['loss_contrast']:.6f}"),
        ('Runtime', f"{test_info['run_time']}")
    )
    logging.info('\n'.join((
        '',
        center_text(f"Contrastive Test - {config['epochs']:4}", ' '),
        table
        )))



    test_info = evaluate(test_data, best_model, config['epochs'] + 1)
    table = make_table(
        ('Total loss', f"{test_info['loss']:.6f}"),
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
        wandb.log({"Test Loss" : test_info['loss']})
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
