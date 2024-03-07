"""Dataset specification for hit graphs using pytorch_geometric formuation"""

# System imports
import os

# External imports
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, random_split
import torch_geometric
import random

from collections import namedtuple
import dataclasses

@dataclasses.dataclass
class EventInfo:
    hit_cartesian: np.ndarray
    hit_cylindrical: np.ndarray
    layer_id: np.ndarray
    n_pixels: np.ndarray
    energy: np.ndarray
    momentum: np.ndarray
    interaction_point: np.ndarray
    trigger: np.ndarray
    has_trigger_pair: np.ndarray
    track_origin: np.ndarray
    edge_index: np.ndarray
    edge_z0: np.ndarray
    edge_phi_slope: np.ndarray
    phi_slope_max: float
    z0_max: float
    trigger_node: np.ndarray
    particle_id: np.ndarray
    particle_type: np.ndarray
    parent_particle_type: np.ndarray
    cylindrical_std: np.ndarray
    hit_type: np.ndarray

def calc_dphi(phi1, phi2):
    """Computes phi2-phi1 given in range [-pi,pi]"""
    dphi = phi2 - phi1
    dphi[dphi > np.pi] -= 2*np.pi
    dphi[dphi < -np.pi] += 2*np.pi
    return dphi


def build_edges(event_info, phi_slope_max, z0_max):
    r, phi, z  = event_info.hit_cylindrical.T
    layer_pairs = [(0,1), (1,2), (2,3), (3,4), (4,5), (5,6), (1, 3), (1,4), (2,4), (2,5), (3,5), (3,6), (4,6)]
    hit_ids = np.arange(len(event_info.hit_cylindrical))
    edge_candidates = []
    phi_slopes = []
    z0s = []
    layer_id = event_info.layer_id
    for (layer1, layer2) in layer_pairs:
        mask1 = layer_id == layer1
        mask2 = layer_id == layer2
        if np.sum(mask1) == 0 or np.sum(mask2) == 0:
            continue
        h1 = hit_ids[mask1]
        h2 = hit_ids[mask2]
        edges = np.stack(np.meshgrid(h1, h2, indexing='xy'), axis=-1).reshape(-1, 2)

        z1 = z[mask1]
        z2 = z[mask2]
        r1 = r[mask1]
        r2 = r[mask2]
        phi1 = phi[mask1]
        phi2 = phi[mask2]

        dphi = calc_dphi(phi2.reshape(-1, 1), phi1.reshape(1, -1))
        dr = r2.reshape(-1, 1) - r1.reshape(1, -1)
        dz = z2.reshape(-1, 1) - z1.reshape(1, -1)
        phi_slope = dphi / dr
        z0 = z1 - r1 * dz / dr
        good_seg_mask = (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)
        good_seg_mask = good_seg_mask.reshape(-1)
        edge_candidates.append(edges[good_seg_mask])
        phi_slopes.append(phi_slope.reshape(-1)[good_seg_mask])
        z0s.append(z0.reshape(-1)[good_seg_mask])

    return np.concatenate(edge_candidates, axis=0).T, np.concatenate(phi_slopes), np.concatenate(z0s)

def load_file(filename):
    with np.load(filename, allow_pickle=True) as f:
        return EventInfo(
                hit_cartesian=f['hit_cartesian'],
                hit_cylindrical=f['hit_cylindrical'],
                layer_id=f['layer_id'],
                n_pixels=f['n_pixels'],
                energy=f['energy'],
                momentum=f['momentum'],
                interaction_point=f['interaction_point'],
                trigger=f['trigger'],
                has_trigger_pair=f['has_trigger_pair'],
                track_origin=f['track_origin'],
                edge_index=f['edge_index'],
                edge_z0=f['edge_z0'],
                edge_phi_slope=f['edge_phi_slope'],
                phi_slope_max=f['phi_slope_max'],
                z0_max=f['z0_max'],
                trigger_node=f['trigger_node'],
                particle_id=f['particle_id'],
                particle_type=f['particle_type'],
                parent_particle_type=f['parent_particle_type'],
                cylindrical_std=f['cylindrical_std'],
                hit_type=f['hit_type']
        )




def load_graph(filename, cylindrical_features_scale, phi_slope_max, z0_max, use_intt, use_cylindrical_std, use_hit_type):
    event_info = load_file(filename)

    if not use_intt:
        keep = event_info.layer_id <= 2
    else:
        keep = np.ones(event_info.layer_id.shape[0], dtype=bool)


    x = np.concatenate([
        event_info.hit_cylindrical/cylindrical_features_scale[None],
        event_info.n_pixels.reshape(-1, 1), 
        event_info.layer_id.reshape(-1, 1),
    ], axis=-1)

    if use_cylindrical_std:
        x = np.concatenate([x, event_info.cylindrical_std], axis=-1)

    if use_hit_type:
        x = np.concatenate([x, event_info.hit_type.reshape(-1, 1)], axis=-1)
        layer_id = event_info.layer_id
        # INTT hits
        x[layer_id > 2, 4] = 0



    edge_index = event_info.edge_index
    phi_slope = event_info.edge_phi_slope
    z0 = event_info.edge_z0
    edge_index = edge_index[:, (np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)]
    pid = event_info.particle_id
    keep_edge = keep[edge_index[0]] & keep[edge_index[1]]

    edge_index = edge_index[:, keep_edge]

    y = event_info.trigger

    event_info.edge_index = edge_index

    event_info.edge_phi_slope = (phi_slope[(np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)])[keep_edge]
    event_info.edge_z0 = (z0[(np.abs(phi_slope) <= phi_slope_max) & (np.abs(z0) <= z0_max)])[keep_edge]
    event_info.phi_slope_max = phi_slope_max
    event_info.z0_max = z0_max


    return x, edge_index, y, event_info



class HitGraphDataset(Dataset):
    """PyTorch dataset specification for hit graphs"""

    def __init__(self, input_dir=None, filelist=None, n_samples=None, real_weight=1.0, n_folders=1, input_dir2=None, phi_slope_max=0.03, z0_max=200, cylindrical_features_scale=(3, 1, 3), load_full_event=False, use_cylindrical_std=False, use_hit_type=False):
        self.load_full_event = load_full_event
        self.cylindrical_features_scale = np.array(cylindrical_features_scale)
        if filelist is not None:
            self.metadata = pd.read_csv(os.path.expandvars(filelist))
            filenames = self.metadata.file.values
        elif input_dir is not None:
            input_dir = os.path.expandvars(input_dir)
            filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
        else:
            raise Exception('Must provide either input_dir or filelist to HitGraphDataset')

        self.filenames = filenames if n_samples is None else filenames[:n_samples]
        if n_folders == 2:
            filenames = sorted([os.path.join(input_dir2, f) for f in os.listdir(input_dir2)
                                if f.startswith('event') and not f.endswith('_ID.npz')])

            self.filenames += filenames[:n_samples]
        self.real_weight = real_weight
        self.fake_weight = 1 #real_weight / (2 * real_weight - 1)
        self.phi_slope_max = phi_slope_max
        self.z0_max = z0_max
        self.use_hit_type = use_hit_type
        self.use_cylindrical_std = use_hit_type


    def __getitem__(self, index):
        event_file_name = self.filenames[index]
        x, edge_index, y, event_info = load_graph(self.filenames[index], self.cylindrical_features_scale, self.phi_slope_max, self.z0_max, use_intt=True, use_cylindrical_std=self.use_cylindrical_std, use_hit_type=self.use_hit_type)
            
        if self.load_full_event:
            return torch_geometric.data.Data(
                    x=torch.from_numpy(x).to(torch.float),
                    edge_index=torch.from_numpy(edge_index).to(torch.long),
                    trigger=torch.from_numpy(y).to(torch.long), 
                    w=torch.from_numpy(w).to(torch.float),
                    i=index, 
                    filename=event_file_name,
                    event_info=event_info
            )
        else:
            return torch_geometric.data.Data(
                    x=torch.from_numpy(x).to(torch.float),
                    edge_index=torch.from_numpy(edge_index).to(torch.long),
                    trigger=torch.from_numpy(y).to(torch.long), 
                    i=index, 
                    filename=event_file_name,
            )
       
    def __len__(self):
        return len(self.filenames)

def get_datasets(n_train, n_valid, input_dir=None, filelist=None, real_weight=1.0, n_folders=1, input_dir2=None, phi_slope_max=0.03, z0_max=200, load_full_event=False, use_cylindrical_std=False, use_hit_type=False):
    data = HitGraphDataset(input_dir=input_dir, filelist=filelist,
                           n_samples=n_train+n_valid, real_weight=real_weight, n_folders=n_folders, input_dir2=input_dir2, phi_slope_max=phi_slope_max, z0_max=z0_max, load_full_event=load_full_event, use_cylindrical_std=use_cylindrical_std, use_hit_type=use_hit_type)

    if n_folders == 1:
        train_data, valid_data = random_split(data, [n_train, n_valid])
    if n_folders == 2:
        train_data, valid_data = random_split(data, [2*n_train, 2*n_valid])
    return train_data, valid_data
