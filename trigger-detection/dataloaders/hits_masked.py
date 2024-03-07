from collections import namedtuple
# System imports
import os
import random

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split, Sampler
import torch_geometric
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data.dataloader import default_collate
import tqdm

from numpy.linalg import inv
from icecream import ic
from collections import namedtuple

BatchInfo = namedtuple('BatchInfo',
        'track_vector trigger n_tracks is_trigger_track momentums ip'
)
EventInfo = namedtuple('EventInfo',
        'hits origin_vertices momentums pids is_trigger_track ptypes ip trigger'
)

def load_graph(filename, load_complete_graph=False):
    with np.load(filename, allow_pickle=True) as f:
        complete_flags = f['is_complete_trk'].astype(bool)
        if load_complete_graph and len(complete_flags) != 0:
            hits = f['hits_xyz'][complete_flags]
            origin_vertices = f['psv'][complete_flags]
            momentums = f['p_momentum'][complete_flags]
            pids = f['pid'][complete_flags]
            is_trigger_track = f['trigger_track_flag'][complete_flags].astype(bool)
            ptypes = f ['ParticleTypeID'][complete_flags]
        else:
            hits = f['hits_xyz']
            origin_vertices = f['psv']
            momentums = f['p_momentum']
            pids = f['pid']
            is_trigger_track = f['trigger_track_flag'].astype(bool)
            ptypes = f ['ParticleTypeID']


        trigger = f['trigger']
        ip = f['ip']

    return EventInfo(
            hits=hits, 
            origin_vertices=origin_vertices, 
            momentums=momentums, 
            pids=pids, 
            is_trigger_track=is_trigger_track, 
            ptypes=ptypes, 
            trigger=trigger, 
            ip=ip, 
        )

class TrackDataset(object):
    """PyTorch dataset specification for hit graphs"""

    def __init__(
            self, 
            trigger_input_dir, 
            nontrigger_input_dir, 
            n_trigger_samples,
            n_nontrigger_samples,
            load_complete_graph=False,
            add_geo_features=True,
            use_radius=True,
            use_predicted_pz=False,
            use_momentum=False,
            use_trigger=True,
            use_nontrigger=True):
        self.filenames = []
        if use_trigger:
            input_dir = os.path.expandvars(trigger_input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            random.shuffle(filenames)
            self.filenames = filenames[:n_trigger_samples]

        if use_nontrigger:
            input_dir = os.path.expandvars(nontrigger_input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                            if f.startswith('event') and not f.endswith('_ID.npz')])
            self.filenames += filenames[:n_nontrigger_samples]
            random.shuffle(self.filenames)
       
        self.load_complete_graph = load_complete_graph
        self.use_momentum = use_momentum

    def __getitem__(self, file_index):
        event_info = load_graph(self.filenames[file_index], self.load_complete_graph)
        hits = event_info.hits


        if self.use_momentum and hits.shape[0] != 0:
            hits = np.concatenate([hits, event_info.momentums], axis=-1)
        elif self.use_momentum:
            hits = np.concatenate([hits, np.zeros((0, 3))], axis=-1)

        n_hits = hits.shape[0]
        if hits.shape[0] == 0:
            momentums = np.zeros((0, 3))
        else:
            momentums = []
            for m in event_info.momentums:
                if m is None:
                    momentums.append([0, 0, 0])
                else:
                    momentums.append(m)
            momentums = np.stack(momentums, axis=0)

        return BatchInfo(
                track_vector=torch.from_numpy(hits).to(torch.float),
                n_tracks=hits.shape[0],
                trigger=torch.from_numpy(event_info.trigger),
                is_trigger_track=torch.from_numpy(event_info.is_trigger_track),
                momentums=torch.from_numpy(momentums),
                ip=torch.from_numpy(event_info.ip)
            )

    def __len__(self):
        return len(self.filenames)


def get_datasets(n_train, n_valid, n_test, 
        trigger_input_dir=None, 
        nontrigger_input_dir=None,
        load_complete_graph=False, 
        use_momentum=False,
        use_trigger=True,
        use_nontrigger=True):
    data = TrackDataset(trigger_input_dir=trigger_input_dir,
                        nontrigger_input_dir=nontrigger_input_dir,
                        n_trigger_samples=n_train+n_valid+n_test,
                        n_nontrigger_samples=n_train+n_valid+n_test,
                        load_complete_graph=load_complete_graph,
                        use_momentum=use_momentum,
                        use_trigger=use_trigger,
                        use_nontrigger=use_nontrigger)

    total = use_trigger + use_nontrigger
    train_data, valid_data, test_data = random_split(data, [total*n_train, total*n_valid, total*n_test])

    return train_data, valid_data, test_data
