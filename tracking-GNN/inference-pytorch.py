from dataclasses import replace
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import os.path
import sys
import logging
import tqdm
import pickle
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from models.agnn_inference import GNNSegmentClassifier
from icecream import ic
from numpy.linalg import inv
import sklearn.metrics as metrics
from datasets import get_data_loaders
import yaml

import dataclasses

is_check_acc = True
is_save = True
DEVICE = 'cuda:1'

if is_save:
    output_dirs = ['tracking-sweep-0/wordly-sweep-192/nontrigger/0/', 'tracking-sweep-0/wordly-sweep-192/trigger/1/']
    for output_dir in output_dirs:
        os.makedirs(output_dir, exist_ok=True)

model_result_folder = 'tracking_results/agnn/agnn-lr0.006905309379469133-b512-d8-ReLU-gi1-ln-False-n1000000/experiment_2023-07-11_09:07:40'



# create model and load checkpoint
config_file = model_result_folder + '/config.pkl'
config = pickle.load(open(config_file, 'rb'))

data_config = config.get('data')
data_config['load_full_event'] = True
data_config['batch_size'] = 36
data_config['load_all'] = True
train_data_loader, valid_data_loader = get_data_loaders(
    distributed=False, rank=0, n_ranks=1, **data_config)



model_config = config.get('model', {})
model_config.pop('loss_func')
model_config.pop('name')
model = GNNSegmentClassifier(**model_config).to(DEVICE)

def load_checkpoint(checkpoint_file, model, optimizer=None):
    checkpoint = torch.load(checkpoint_file, map_location=torch.device('cpu'))
    model.load_state_dict(checkpoint['model'])
    if optimizer != None:
        optimizer.load_state_dict(checkpoint['optimizer'])
        return model, optimizer
    return model

# load_checkpoint
checkpoint_dir = os.path.join(model_result_folder, 'checkpoints')
checkpoint_file = sorted([os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith('model_checkpoint')])
summary = np.genfromtxt(os.path.join(model_result_folder, 'summaries_0.csv'), delimiter=',', skip_header=1)
f1 = summary[:, 5]
best_epoch = np.argmax(f1)
print(f'Best epoch: {best_epoch}')
checkpoint_file = checkpoint_file[best_epoch]
print(checkpoint_file)
model = load_checkpoint(checkpoint_file, model)
print('Successfully reloaded!')

sigmoid = torch.nn.Sigmoid()

if is_check_acc:
    preds = []
    labels = []

for data_loader in [train_data_loader, valid_data_loader]:
    for i, batch in enumerate(tqdm.tqdm(data_loader, smoothing=0.0)):
        x = batch.x.to(DEVICE)
        b = batch.batch.to(DEVICE)
        edge_index = batch.edge_index.to(DEVICE)

        all_model_edge_probability = sigmoid(model((x, edge_index)))
        for j in range(len(batch.event_info)):
            model_edge_probability = all_model_edge_probability[b[edge_index[0]] == j].detach().cpu().numpy().reshape(-1)

            if is_check_acc:
                start, end = batch.event_info[j].edge_index

                pid = batch.event_info[j].particle_id
                # all track
                label = pid[start] == pid[end]

                # pos track
                # label = pid[start] == pid[end]

                # high momentum track
                # momentum = p_momentum
                # momentum_mask = [m is not None and m[0] ** 2 + m[1] ** 2 > 0.04 for m in momentum[start]]
                # label = np.logical_and(np.logical_and(pid[start] != -1, pid[start] == pid[end]), momentum_mask)
                
                labels.append(label)
                preds.append(model_edge_probability.reshape(-1)>0.5)
            
            if is_save:
                event = dataclasses.asdict(batch.event_info[j])
                event['model_edge_probability'] = model_edge_probability
                filename = batch.filename[j]

                output_dir = output_dirs[event['trigger']]
                out_filename = os.path.join(output_dir, filename.split('/')[-1])
                np.savez(out_filename, **event)


if is_check_acc:
    labels = np.hstack(labels)
    preds = np.hstack(preds)

    result = {'prec': metrics.precision_score(labels, preds>0),
                'recall': metrics.recall_score(labels, preds>0),
                'acc': metrics.accuracy_score(labels, preds>0),
                'F1': metrics.f1_score(labels, preds>0)}

    print(result)
