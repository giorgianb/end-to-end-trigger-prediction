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
import functools
from typing import Union

from numpy.linalg import inv
from icecream import ic
from collections import namedtuple
from disjoint_set import DisjointSet
import dataclasses
from scipy.stats import mode
from . import utils
from . import tracks
from . import pred_tracks
PERCENTILES = [0.0, 0.07736333, 0.15935145, 0.23861714, 0.31776543, 0.39706996, 0.47612439, 0.55537474, 0.63460629, 0.7137, 0.792404, 0.87207085, 0.9, 0.92323239, 1.00635142, 1.0916517, 1.15766312, 1.21524116, 1.28409604, 1.36827297, 1.4769098, 1.55696555, 1.62481218, 1.69213715, 1.75821458, 1.82453213, 1.89061091, 1.9563535, 2.02146734, 2.08087522, 2.13369944, 2.18624562, 2.24261664, 2.30021808, 2.36522198, 2.43604802, 2.45331804, 2.45666897, 2.5, 2.5, 2.56519136, 2.64685111, 2.7135639, 2.78241047, 2.84623904, 2.90785439, 2.9664565, 3.03816257, 3.12106569, 3.19993773, 3.2375626, 3.2768973, 3.32030477, 3.40299416, 3.51451752, 3.6397848, 3.72478132, 3.80094705, 3.898048, 4.00926403, 4.07522167, 4.1, 4.12611999, 4.2127149, 4.37240229, 4.74167553, 5.01529627, 5.248522, 5.48321202, 5.7, 5.7, 5.84536892, 5.99177291, 6.14549803, 6.30729265, 6.46952291, 6.65282291, 6.96292396, 7.1844, 7.23660151, 7.3, 7.36109355, 7.50240874, 7.67905274, 8.1243705, 8.49908919, 8.74405354, 8.9, 8.9443053, 9.20377323, 9.6764, 9.91596494, 10.12492878, 10.5, 10.73250401, 11.90454072, 12.1, 14.1, 16.1, 18.1, 22.1]

class EpochDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epoch = 0

    def __iter__(self):
        self.dataset.dataset.epoch = self.epoch
        for batch in super().__iter__():
            yield batch
        self.epoch += 1

@dataclasses.dataclass
class EventInfo:
    n_pixels: Union[np.ndarray, torch.Tensor]
    energy: Union[np.ndarray, torch.Tensor]
    momentum: Union[np.ndarray, torch.Tensor]
    interaction_point: Union[np.ndarray, torch.Tensor]
    trigger: Union[bool, torch.Tensor]
    has_trigger_pair: Union[bool, torch.Tensor]
    track_origin: Union[np.ndarray, torch.Tensor]
    #trigger_node: Union[np.ndarray, torch.Tensor]
    particle_id: Union[np.ndarray, torch.Tensor]
    particle_type: Union[np.ndarray, torch.Tensor]
    parent_particle_type: Union[np.ndarray, torch.Tensor]
    track_hits: Union[np.ndarray, torch.Tensor]
    track_n_hits: Union[np.ndarray, torch.Tensor]

@dataclasses.dataclass
class BatchInfo(EventInfo):
    track_vector: torch.Tensor



def get_tracks(edge_index):
    # Get connected components
    ds = DisjointSet()
    for i in range(edge_index.shape[1]):
        ds.union(edge_index[0, i], edge_index[1, i])

    return tuple(list(x) for x in ds.itersets())

class TrackDataset(object):
    """PyTorch dataset specification for hit graphs"""

    def __init__(
            self, 
            gt_trigger_input_dir, 
            gt_nontrigger_input_dir, 
            pred_trigger_input_dir,
            pred_nontrigger_input_dir,
            n_trigger_samples,
            n_nontrigger_samples,
            min_edge_probability=0.5,
            intt_required = False,
            use_geometric_features=False,
            use_radius=False,
            use_center=False,
            use_predicted_pz=False,
            use_momentum=False,
            use_transverse_momentum=False,
            use_parallel_momentum=False,
            use_energy=False,
            use_n_hits=False,
            use_n_pixels=False,
            use_cylindrical_std=False,
            use_hit_type=False,
            n_hit_type=1,
            rescale_by_percentile=-1,
            percentiles=PERCENTILES,
            n_epochs=16,
            ):
        self.filenames = []
        if gt_trigger_input_dir is not None:
            input_dir = os.path.expandvars(gt_trigger_input_dir)
            gt_filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            random.shuffle(gt_filenames)
            self.gt_filenames = gt_filenames[:n_trigger_samples]

        if gt_nontrigger_input_dir is not None:
            input_dir = os.path.expandvars(gt_nontrigger_input_dir)
            gt_filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                            if f.startswith('event') and not f.endswith('_ID.npz')])
            self.gt_filenames += gt_filenames[:n_nontrigger_samples]
            random.shuffle(self.gt_filenames)

        if pred_trigger_input_dir is not None:
            input_dir = os.path.expandvars(pred_trigger_input_dir)
            pred_filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                                if f.startswith('event') and not f.endswith('_ID.npz')])
            random.shuffle(pred_filenames)
            self.pred_filenames = pred_filenames[:n_trigger_samples]

        if pred_nontrigger_input_dir is not None:
            input_dir = os.path.expandvars(pred_nontrigger_input_dir)
            pred_filenames = sorted([os.path.join(input_dir, f) for f in os.listdir(input_dir)
                            if f.startswith('event') and not f.endswith('_ID.npz')])
            self.pred_filenames += pred_filenames[:n_nontrigger_samples]
            random.shuffle(self.pred_filenames)

        if 0 <= rescale_by_percentile <= 100:
            self.rescale_factor = percentiles[rescale_by_percentile]
        else:
            self.rescale_factor = 1
       
        self.use_geometric_features = use_geometric_features
        self.use_radius = use_radius
        self.use_center = use_center
        self.use_predicted_pz = use_predicted_pz
        self.use_momentum = use_momentum
        self.use_transverse_momentum = use_transverse_momentum
        self.use_parallel_momentum = use_parallel_momentum
        self.use_energy = use_energy
        self.use_n_hits = use_n_hits
        self.use_n_pixels = use_n_pixels
        self.use_cylindrical_std = use_cylindrical_std
        self.use_hit_type = use_hit_type
        self.n_hit_type = n_hit_type
        self.min_edge_probability = min_edge_probability
        self.n_epochs = n_epochs
        self.epoch = 0

    def __getitem__(self, file_index):
        if random.random() >= self.epoch/self.n_epochs:
            event_info = tracks.load_graph(self.gt_filenames[file_index])
        else:
            event_info = pred_tracks.load_graph(self.pred_filenames[file_index], min_edge_probability=self.min_edge_probability)
            
        track_vector = event_info.track_hits / self.rescale_factor

        if self.use_geometric_features:
            geo_features = utils.calculate_geometric_features(event_info.track_hits / self.rescale_factor)
            track_vector = np.concatenate([track_vector, geo_features], axis=-1)

        if self.use_radius or self.use_center or self.use_predicted_pz:
            good_layers = np.any(event_info.track_hits.reshape(-1, 5, 3) != 0, axis=-1)
            n_layers = np.sum(good_layers, axis=-1)
            radius, center = utils.get_approximate_radii(event_info.track_hits, good_layers, n_layers)

        if self.use_radius:
            r = radius / self.rescale_factor
            track_vector = np.concatenate([track_vector, r[..., None]], axis=-1)

        if self.use_center:
            c = center / self.rescale_factor
            track_vector = np.concatenate([track_vector, c], axis=-1)

        if self.use_momentum:
            track_vector = np.concatenate([track_vector, event_info.momentum], axis=-1)

        if self.use_energy:
            track_vector = np.concatenate([track_vector, event_info.energy[..., None]], axis=-1)


        if self.use_transverse_momentum:
            p_t = np.sqrt(np.sum(event_info.momentum[:, :2]**2, axis=-1))[..., None]
            track_vector = np.concatenate([track_vector, p_t], axis=-1)

        if self.use_parallel_momentum and track_vector.shape[0] != 0:
            p_z = event_info.momentum[:, 2][..., None]
            track_vector = np.concatenate([track_vector, p_z], axis=-1)

        if self.use_predicted_pz:
            pred_pz = utils.get_predicted_pz(event_info.track_hits / self.rescale_factor, 
                    good_layers, 
                    radius / self.rescale_factor
            )
            track_vector = np.concatenate([track_vector, pred_pz[..., None]], axis=-1)

        if self.use_n_hits:
            track_vector = np.concatenate([track_vector, event_info.track_n_hits], axis=-1)

        if self.use_n_pixels:
            track_vector = np.concatenate([track_vector, event_info.n_pixels], axis=-1)

        #if track_vector.shape[0] != 0:
        #    hit_type = event_info.hit_type[:, :, :self.n_hit_type].reshape(track_vector.shape[0], -1)
        #    cylindrical_std = event_info.cylindrical_std.reshape(track_vector.shape[0], -1)
        #else:
        #    cylindrical_std = np.zeros((0, 15))
        #    hit_type = np.zeros((0, 5*self.n_hit_type))

        #if self.use_cylindrical_std:
        #    track_vector = np.concatenate([track_vector, cylindrical_std], axis=-1)

        #if self.use_hit_type:
        #    track_vector = np.concatenate([track_vector, hit_type], axis=-1)


        return BatchInfo(
                track_vector=track_vector.astype(np.float32),
                n_pixels=event_info.n_pixels.astype(np.float32),
                energy=event_info.energy[:, None].astype(np.float32),
                momentum=event_info.momentum.astype(np.float32),
                interaction_point=event_info.interaction_point.astype(np.float32),
                trigger=event_info.trigger.astype(np.float32),
                has_trigger_pair=event_info.has_trigger_pair.astype(np.float32),
                track_origin=event_info.track_origin.astype(np.float32),
                #trigger_node=event_info.trigger_node.astype(np.float32),
                particle_id=event_info.particle_id.astype(np.float32),
                particle_type=event_info.particle_type.astype(np.float32),
                parent_particle_type=event_info.parent_particle_type.astype(np.float32),
                track_hits=event_info.track_hits.astype(np.float32),
                track_n_hits=event_info.track_n_hits.astype(np.float32),
                #cylindrical_std=cylindrical_std.astype(np.float32),
                #hit_type=hit_type.astype(np.float32)
            )

    def __len__(self):
        return len(self.gt_filenames)


def get_datasets(n_train, n_valid, n_test, 
        gt_trigger_input_dir=None, 
        gt_nontrigger_input_dir=None,
        pred_trigger_input_dir=None, 
        pred_nontrigger_input_dir=None,
        min_edge_probability=0.5,
        intt_required=False,
        use_geometric_features=False,
        use_radius=False,
        use_center=False,
        use_predicted_pz=False,
        use_momentum=False,
        use_transverse_momentum=False,
        use_parallel_momentum=False,
        use_energy=False,
        use_n_pixels=False,
        use_n_hits=False,
        rescale_by_percentile=-1,
        use_cylindrical_std=False,
        use_hit_type=False,
        n_hit_type=1,
        percentiles=PERCENTILES,
        tmp_dir=None,
        n_epochs=16):
    data = TrackDataset(gt_trigger_input_dir=gt_trigger_input_dir,
                        gt_nontrigger_input_dir=gt_nontrigger_input_dir,
                        pred_trigger_input_dir=pred_trigger_input_dir,
                        pred_nontrigger_input_dir=pred_nontrigger_input_dir,
                        n_trigger_samples=n_train+n_valid+n_test,
                        n_nontrigger_samples=n_train+n_valid+n_test,
                        min_edge_probability=min_edge_probability,
                        intt_required=intt_required,
                        use_geometric_features=use_geometric_features,
                        use_radius=use_radius,
                        use_center=use_center,
                        use_predicted_pz=use_predicted_pz,
                        use_momentum=use_momentum,
                        use_transverse_momentum=use_transverse_momentum,
                        use_parallel_momentum=use_parallel_momentum,
                        use_energy=use_energy,
                        use_n_pixels=use_n_pixels,
                        use_n_hits=use_n_hits,
                        rescale_by_percentile=-1,
                        use_cylindrical_std=use_cylindrical_std,
                        use_hit_type=use_hit_type,
                        n_hit_type=n_hit_type,
                        percentiles=PERCENTILES)

    total = (gt_trigger_input_dir is not None) + (gt_nontrigger_input_dir is not None)
    train_data, valid_data, test_data = random_split(data, [total*n_train, total*n_valid, total*n_test])

    return train_data, valid_data, test_data
