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

EventInfo = namedtuple('EventInfo',
        'track_vector complete_flags origin_vertices momentums pids is_trigger_track ptypes energy trigger ip adj r physics_pred'
)

BatchInfo = namedtuple('BatchInfo',
        'track_vector trigger n_tracks is_trigger_track momentums energies'
)

def load_event(filename, add_geo_features, use_radius, use_momentum, use_energy, use_transverse_momentum, use_parallel_momentum, use_predicted_pz, load_complete_graph=False):
    event_info = load_graph(filename, load_complete_graph)
    track_vector = event_info.track_vector
    if track_vector.shape[0] != 0 and add_geo_features:
        # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
        geo_features = np.zeros((track_vector.shape[0], 13))
        phi  = np.zeros((track_vector.shape[0], 5))
        geo_features[:, 5] = np.arctan2(track_vector[:, 1], track_vector[:, 0])
        for i in range(4):
            geo_features[:, i] = get_length(
                    track_vector[:, (3*i+3):(3*i+6)], 
                    track_vector[:, (3*i):(3*i+3)]
                    )
        for i in range(5):
            phi[:, i] = np.arctan2(
                    track_vector[:, (3*i)+1], 
                    track_vector[:, (3*i)]
                    )
        geo_features[:, 5] = get_length(
                track_vector[:, 12:15], 
                track_vector[:, 0:3]
                )
        geo_features[:, 6:10] = np.diff(phi)
        geo_features[:, 10:13] = np.mean(
                track_vector.reshape((track_vector.shape[0], 5, 3)), axis=(0, 1)
                )
        track_vector = np.concatenate([track_vector, geo_features], axis=1)
    elif add_geo_features:
        track_vector = np.concatenate([track_vector, np.zeros((0, 13))], axis=-1)

    if use_radius and track_vector.shape[0] != 0:
        r = event_info.r
        track_vector = np.concatenate([track_vector, r], axis=-1)
    elif use_radius:
        track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

    if use_momentum and track_vector.shape[0] != 0:
        track_vector = np.concatenate([track_vector, event_info.momentums], axis=-1)
    elif use_momentum:
        track_vector = np.concatenate([track_vector, np.zeros((0, 3))], axis=-1)

    if use_energy and track_vector.shape[0] != 0:
        track_vector = np.concatenate([track_vector, np.expand_dims(event_info.energy, -1)], axis=-1)
    elif use_energy:
        track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)


    if use_transverse_momentum and track_vector.shape[0] != 0:
        p_t = np.expand_dims(np.sqrt(np.sum(event_info.momentums[:, :2]**2, axis=-1)), -1)
        track_vector = np.concatenate([track_vector, p_t], axis=-1)
    elif use_transverse_momentum:
        track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

    if use_parallel_momentum and track_vector.shape[0] != 0:
        p_z = np.expand_dims(event_info.momentums[:, 2], -1)
        track_vector = np.concatenate([track_vector, p_z], axis=-1)
    elif use_parallel_momentum:
        track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

    if use_predicted_pz and track_vector.shape[0] != 0:
        hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
        good_hits = np.any(hits != 0, axis=-1)
        first_hit, last_hit = get_track_endpoints(hits, good_hits)
        pred_pz = get_predicted_pz(first_hit, last_hit, event_info.r.reshape(-1))
        track_vector = np.concatenate([track_vector, np.expand_dims(pred_pz, -1)], axis=-1)
    elif use_predicted_pz:
        track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

    n_tracks = track_vector.shape[0]
    if track_vector.shape[0] == 0:
        momentums = np.zeros((0, 3))
    else:
        momentums = event_info.momentums

    return BatchInfo(
            track_vector=torch.from_numpy(track_vector).to(torch.float),
            n_tracks=n_tracks,
            trigger=torch.from_numpy(event_info.trigger),
            is_trigger_track=torch.from_numpy(event_info.is_trigger_track),
            momentums=torch.from_numpy(momentums),
            # TODO: fix ugly hack
            energies=torch.ones(1)
        )


def get_track_endpoints(hits, good_hits):
    # Assumption: all tracks have at least 1 hit
    # If it has one hit, first_hit == last_hit for that track
    # hits shape: (n_tracks, 5, 3)
    # good_hits shape: (n_tracks, 5)
    min_indices = good_hits * np.arange(5) + (1 - good_hits) * np.arange(5, 10)
    indices = np.expand_dims(np.argmin(min_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    first_hits = np.take_along_axis(hits, indices, axis=-2)
    max_indices = good_hits * np.arange(5, 10) + (1 - good_hits) * np.arange(5)
    indices = np.expand_dims(np.argmax(max_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    last_hits = np.take_along_axis(hits, indices, axis=-2)
    return first_hits.squeeze(1), last_hits.squeeze(1)

def get_predicted_pz(first_hit, last_hit, radius):
    dz = (last_hit[:, -1] - first_hit[:, -1])/100
    chord2 = ((last_hit[:, 0] - first_hit[:, 0]) ** 2 + (last_hit[:, 1] - first_hit[:, 1]) ** 2) / 10000
    with np.errstate(invalid='ignore'):
        dtheta = np.arccos((2*radius**2 - chord2) / (2*radius**2 + 1e-10))
    return np.nan_to_num(dz / dtheta)

def load_graph(filename, load_complete_graph=False):
    with np.load(filename) as f:
        complete_flags = f['complete_flags'] 
        if load_complete_graph and len(complete_flags) != 0:
            track_vector = f['track_vector'][complete_flags]
            origin_vertices = f['origin_vertices'][complete_flags]
            momentums = f['momentums'][complete_flags]
            pids = f['pids'][complete_flags]
            is_trigger_track = f['trigger_track_flags'][complete_flags]
            ptypes = f ['ptypes'][complete_flags]
            energy = f['energy'][complete_flags]
            r = f['r'][complete_flags]

            if 'physics_pred' in f.keys():
                physics_pred = f['physics_pred'][complete_flags]
            else:
                physics_pred = np.zeros((track_vector.shape[0], 0))

        else:
            track_vector = f['track_vector']
            origin_vertices = f['origin_vertices']
            momentums = f['momentums']
            pids = f['pids']
            is_trigger_track = f['trigger_track_flags']
            ptypes = f ['ptypes']
            energy = f['energy']
            r = f['r']

            if 'physics_pred' in f.keys():
                physics_pred = f['physics_pred']
            else:
                physics_pred = np.zeros((track_vector.shape[0], 0))

        trigger = f['trigger']
        ip = f['ip']
        n_track = track_vector.shape[0]
        if n_track != 0:
            adj = (origin_vertices[:,None] == origin_vertices).all(axis=2)
        else:
            adj = np.array([[]])


    return EventInfo(
            track_vector=track_vector, 
            complete_flags=complete_flags, 
            origin_vertices=origin_vertices, 
            momentums=momentums, 
            pids=pids, 
            is_trigger_track=is_trigger_track, 
            ptypes=ptypes, 
            energy=energy, 
            trigger=trigger, 
            ip=ip, 
            adj=adj,
            r=r,
            physics_pred=physics_pred
        )

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))

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
            use_transverse_momentum=False,
            use_parallel_momentum=False,
            use_physics_pred=False,
            use_energy=False,
            use_trigger=True,
            use_nontrigger=True,
            trigger_correction=True):
        self.filenames = []
        self.use_trigger = use_trigger
        self.use_nontrigger = use_nontrigger
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
        self.add_geo_features = add_geo_features
        self.use_radius = use_radius
        self.use_predicted_pz = use_predicted_pz
        self.use_momentum = use_momentum
        self.use_transverse_momentum = use_transverse_momentum
        self.use_parallel_momentum = use_parallel_momentum
        self.use_energy = use_energy
        self.use_physics_pred = use_physics_pred
        self.trigger_correction = True

    def __getitem__(self, file_index):
        event_info = load_graph(self.filenames[file_index//4], self.load_complete_graph)
        track_vector = event_info.track_vector

        n_tracks = track_vector.shape[0]
        track_hits = np.copy(track_vector.reshape(n_tracks, 5, 3))

        if file_index % 4 == 0:
            hits_mask = track_hits[:, :, 0] >= 0
        elif file_index % 4 == 1:
            hits_mask = track_hits[:, :, 0] < 0
        elif file_index % 4 == 2:
            hits_mask = track_hits[:, :, 1] >= 0
        elif file_index % 4 == 3:
            hits_mask = track_hits[:, :, 1] < 0

        track_hits[~hits_mask] = 0
        track_vector = track_hits.reshape(n_tracks, 15)
        empty_tracks = np.all(track_vector == 0, axis=1)
        track_vector = track_vector[~empty_tracks]
        n_trigger_tracks = np.sum(event_info.is_trigger_track[~empty_tracks])
        trigger_change = n_trigger_tracks >= 2
        mask = ~empty_tracks

        if self.use_trigger and self.trigger_correction and not self.use_nontrigger and not trigger_change:
            #ic('starting:', file_index, np.sum(event_info.is_trigger_track))
            for i in range(4):
                track_vector = np.copy(event_info.track_vector)

                n_tracks = track_vector.shape[0]
                track_hits = track_vector.reshape(n_tracks, 5, 3)

                if i % 4 == 0:
                    hits_mask = track_hits[:, :, 0] >= 0
                elif i % 4 == 1:
                    hits_mask = track_hits[:, :, 0] < 0
                elif i % 4 == 2:
                    hits_mask = track_hits[:, :, 1] >= 0
                elif i % 4 == 3:
                    hits_mask = track_hits[:, :, 1] < 0

                track_hits[~hits_mask] = 0
                track_vector = track_hits.reshape(n_tracks, 15)
                empty_tracks = np.all(track_vector == 0, axis=1)
                track_vector = track_vector[~empty_tracks]
                n_trigger_tracks = np.sum(event_info.is_trigger_track[~empty_tracks])
                #ic(np.sum(empty_tracks)/n_tracks)
                #ic(n_trigger_tracks)
                trigger_change = n_trigger_tracks >= 2
                mask = ~empty_tracks

                if trigger_change:
                    break

            #if not trigger_change:
                #ic(file_index, n_trigger_tracks, trigger_change)


        if track_vector.shape[0] != 0 and self.add_geo_features:
            # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
            geo_features = np.zeros((track_vector.shape[0], 13))
            phi  = np.zeros((track_vector.shape[0], 5))
            geo_features[:, 5] = np.arctan2(track_vector[:, 1], track_vector[:, 0])
            for i in range(4):
                geo_features[:, i] = get_length(
                        track_vector[:, (3*i+3):(3*i+6)], 
                        track_vector[:, (3*i):(3*i+3)]
                        )
            for i in range(5):
                phi[:, i] = np.arctan2(
                        track_vector[:, (3*i)+1], 
                        track_vector[:, (3*i)]
                        )
            geo_features[:, 5] = get_length(
                    track_vector[:, 12:15], 
                    track_vector[:, 0:3]
                    )
            geo_features[:, 6:10] = np.diff(phi)
            geo_features[:, 10:13] = np.mean(
                    track_vector.reshape((track_vector.shape[0], 5, 3)), axis=(0, 1)
                    )
            track_vector = np.concatenate([track_vector, geo_features], axis=1)
        elif self.add_geo_features:
            track_vector = np.concatenate([track_vector, np.zeros((0, 13))], axis=-1)

        if self.use_radius and track_vector.shape[0] != 0:
            r = event_info.r[mask]
            track_vector = np.concatenate([track_vector, r], axis=-1)
        elif self.use_radius:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

        if self.use_momentum and track_vector.shape[0] != 0:
            track_vector = np.concatenate([track_vector, event_info.momentums[mask]], axis=-1)
        elif self.use_momentum:
            track_vector = np.concatenate([track_vector, np.zeros((0, 3))], axis=-1)

        if self.use_energy and track_vector.shape[0] != 0:
            track_vector = np.concatenate([track_vector, np.expand_dims(event_info.energy, -1)[mask]], axis=-1)
        elif self.use_energy:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)


        if self.use_transverse_momentum and track_vector.shape[0] != 0:
            p_t = np.expand_dims(np.sqrt(np.sum(event_info.momentums[:, :2]**2, axis=-1)), -1)[mask]
            track_vector = np.concatenate([track_vector, p_t], axis=-1)
        elif self.use_transverse_momentum:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

        if self.use_parallel_momentum and track_vector.shape[0] != 0:
            p_z = np.expand_dims(event_info.momentums[:, 2], -1)[mask]
            track_vector = np.concatenate([track_vector, p_z], axis=-1)
        elif self.use_parallel_momentum:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

        if self.use_predicted_pz and track_vector.shape[0] != 0:
            hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
            good_hits = np.any(hits != 0, axis=-1)
            first_hit, last_hit = get_track_endpoints(hits, good_hits)
            pred_pz = get_predicted_pz(first_hit, last_hit, event_info.r[mask].reshape(-1))
            track_vector = np.concatenate([track_vector, np.expand_dims(pred_pz, -1)], axis=-1)
        elif self.use_predicted_pz:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)

        n_tracks = track_vector.shape[0]
        if track_vector.shape[0] == 0:
            momentums = np.zeros((0, 3))
            energies = np.zeros((0, 1))
        else:
            momentums = event_info.momentums[mask]
            energies = np.expand_dims(event_info.energy, -1)[mask]

        if self.use_physics_pred:
            track_vector = np.concatenate([track_vector, event_info.physics_pred[mask]], axis=-1)

        return BatchInfo(
                track_vector=torch.from_numpy(track_vector).to(torch.float),
                n_tracks=n_tracks,
                trigger=(event_info.trigger & trigger_change),
                is_trigger_track=torch.from_numpy(event_info.is_trigger_track[mask]),
                momentums=torch.from_numpy(momentums),
                energies=torch.from_numpy(energies)
            )

    def __len__(self):
        return 4*len(self.filenames)


def get_datasets(n_train, n_valid, n_test, 
        trigger_input_dir=None, 
        nontrigger_input_dir=None,
        load_complete_graph=False, 
        add_geo_features=True,
        use_radius=True,
        use_predicted_pz=False,
        use_momentum=False,
        use_transverse_momentum=False,
        use_parallel_momentum=False,
        use_physics_pred=False,
        use_energy=False,
        use_trigger=True,
        use_nontrigger=True,
        trigger_correction=True):
    data = TrackDataset(trigger_input_dir=trigger_input_dir,
                        nontrigger_input_dir=nontrigger_input_dir,
                        n_trigger_samples=n_train+n_valid+n_test,
                        n_nontrigger_samples=n_train+n_valid+n_test,
                        load_complete_graph=load_complete_graph,
                        add_geo_features=add_geo_features,
                        use_radius=use_radius,
                        use_predicted_pz=use_predicted_pz,
                        use_momentum=use_momentum,
                        use_transverse_momentum=use_transverse_momentum,
                        use_parallel_momentum=use_parallel_momentum,
                        use_physics_pred=use_physics_pred,
                        use_energy=use_energy,
                        use_trigger=use_trigger,
                        use_nontrigger=use_nontrigger,
                        trigger_correction=trigger_correction)

    total = use_trigger + use_nontrigger
    train_data, valid_data, test_data = random_split(data, [total*4*n_train, total*4*n_valid, total*4*n_test])

    return train_data, valid_data, test_data
