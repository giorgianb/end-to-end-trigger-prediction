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
from . import utils

from numpy.linalg import inv
from icecream import ic
from collections import namedtuple
PERCENTILES = [0.0, 0.07736333, 0.15935145, 0.23861714, 0.31776543, 0.39706996, 0.47612439, 0.55537474, 0.63460629, 0.7137, 0.792404, 0.87207085, 0.9, 0.92323239, 1.00635142, 1.0916517, 1.15766312, 1.21524116, 1.28409604, 1.36827297, 1.4769098, 1.55696555, 1.62481218, 1.69213715, 1.75821458, 1.82453213, 1.89061091, 1.9563535, 2.02146734, 2.08087522, 2.13369944, 2.18624562, 2.24261664, 2.30021808, 2.36522198, 2.43604802, 2.45331804, 2.45666897, 2.5, 2.5, 2.56519136, 2.64685111, 2.7135639, 2.78241047, 2.84623904, 2.90785439, 2.9664565, 3.03816257, 3.12106569, 3.19993773, 3.2375626, 3.2768973, 3.32030477, 3.40299416, 3.51451752, 3.6397848, 3.72478132, 3.80094705, 3.898048, 4.00926403, 4.07522167, 4.1, 4.12611999, 4.2127149, 4.37240229, 4.74167553, 5.01529627, 5.248522, 5.48321202, 5.7, 5.7, 5.84536892, 5.99177291, 6.14549803, 6.30729265, 6.46952291, 6.65282291, 6.96292396, 7.1844, 7.23660151, 7.3, 7.36109355, 7.50240874, 7.67905274, 8.1243705, 8.49908919, 8.74405354, 8.9, 8.9443053, 9.20377323, 9.6764, 9.91596494, 10.12492878, 10.5, 10.73250401, 11.90454072, 12.1, 14.1, 16.1, 18.1, 22.1]

EventInfo = namedtuple('EventInfo',
        'track_vector complete_flags origin_vertices momentums pids is_trigger_track ptypes energy trigger ip adj r center physics_pred n_pixels n_hits'
)

BatchInfo = namedtuple('BatchInfo',
        'track_vector trigger n_tracks is_trigger_track ptypes momentums energies ip origin_vertices'
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

def load_graph(filename, load_compelete_graph=False, phi=0, z=0, noise_std=0):
    # Create rotation matrix
    M = np.array([
        [np.cos(phi), -np.sin(phi)],
        [np.sin(phi), np.cos(phi)]
    ])
        

    with np.load(filename, allow_pickle=True) as f:
        complete_flags = f['is_complete'] 
        if load_compelete_graph and len(complete_flags)!=0:
            track_vector = f['tracks_info'][complete_flags]
            origin_vertices = f['track_2nd_vertex'][complete_flags]
            momentums = f['momentum'][complete_flags].reshape(-1, 3)
            pids = f['pid'] #[complete_flags]
            radius = f['r'][complete_flags]
            c = f['c'][complete_flags]
            is_trigger_track = f['is_trigger_track'][complete_flags]
            ptypes = f ['ParticleTypeID'][complete_flags]
        else:
            track_vector = f['tracks_info']
            if phi != 0:
                hits = track_vector[:, :15].reshape(-1, 5, 3)
                hits_xy = hits[:, :, :2]
                hits_rotated = np.einsum('ij,bkj->bki', M, hits_xy)
                hits = np.concatenate([hits_rotated, hits[:, :, 2:]], axis=-1)
                tracks = hits.reshape(-1, 15)
                track_vector[:, :15] = tracks

            if z != 0:
                hits = track_vector[:, :15].reshape(-1, 5, 3)
                good_hits = np.any(hits, axis=-1)
                hits[:, :, -1] += z*good_hits
                tracks = hits.reshape(-1, 15)
                track_vector[:, :15] = tracks
                track_vector[:, -1] += z

            if noise_std != 0:
                old_hits = np.array(track_vector[:, :15].reshape(-1, 5, 3))
                hits = track_vector[:, :15].reshape(-1, 5, 3)
                good_hits = np.any(hits, axis=-1)
                noise = np.random.normal(0, noise_std, size=hits.shape)
                hits += noise*good_hits[:, :, None]
                tracks = hits.reshape(-1, 15)
                geo_features = utils.calculate_geometric_features(tracks)
                tracks = np.concatenate([tracks, geo_features], axis=-1)


            origin_vertices = f['track_2nd_vertex']
            momentums = f['momentum'].reshape(-1, 3)
            pids = f['pid']
            radius = f['r']
            c = f['c']
            if phi != 0:
                c = np.einsum('ij,kj->ki', M, c)

            if noise_std != 0:
                #good_radius = (radius != 0)
                #radius += np.random.normal(0, 10*noise_std, size=radius.shape) * good_radius
                #c += np.random.normal(0, 10*noise_std, size=c.shape) * good_radius

                n_layers = np.sum(good_hits, axis=-1)
                old_radius, old_c = utils.get_approximate_radii(old_hits.reshape(-1, 15), good_hits, n_layers)
                new_radius, new_c = utils.get_approximate_radii(tracks, good_hits, n_layers)
                r_diff = (new_radius - old_radius)/(old_radius + (old_radius == 0))
                r_diff = r_diff[:, None]
                c_diff = (new_c - old_c)/(old_c + (old_c == 0))
                radius += r_diff * radius
                c += c_diff*c
                #n_layers = np.sum(good_hits, axis=-1)
                #old_radius = radius
                #old_c = c

                #new_radius = new_radius[:, None]
                #old_radius[old_radius == 0] = 1
                #p_r = new_radius / old_radius
                #print(f'{new_radius=} {old_radius=} {p_r=}')
                #t_c = new_c - old_c
                #radius, c = get_approximate_radii(tracks, good_hits, n_layers)
                #radius = radius[:, None]
                #radius *= p_r
                #c += t_c


            is_trigger_track = f['is_trigger_track']
            ptypes = f['ParticleTypeID']
        trigger = f['trigger_flag']
        ip = f['ip']
        n_tracks = f['n_tracks']
        valid_trigger_flag = f['valid_trigger_flag']
        if n_tracks != 0:
            adj = (origin_vertices[:,None] == origin_vertices).all(axis=2)
            adj &= (pids[:,None] != -1) & (pids != -1)
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
            # TODO: fix me
            energy=np.zeros(track_vector.shape[0]),
            trigger=trigger,
            ip=ip,
            r=radius,
            center=c,
            physics_pred=np.zeros((track_vector.shape[0], 0)),
            n_pixels=np.zeros((track_vector.shape[0], 0)),
            n_hits=np.zeros((track_vector.shape[0], 0)),
            adj=adj
        )



def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))

def get_radius(hits):
    def matmul_3D(A, B):
        return np.einsum('lij,ljk->lik', A, B)

    A = np.ones((1, len(hits), 3))
    A[0, :, 0] = hits[:, 0]
    A[0, :, 1] = hits[:, 1]
    y = -(hits[:, 0]**2 + hits[:, 1]**2)
    y = y.reshape((1, y.shape[0], 1))
    AT = np.transpose(A, axes=(0, 2, 1))
    # print(A.shape, AT.shape, y.shape)
    # c = inv(matmul_3D(A, AT))
    c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)
    # print(A.shape, AT.shape, y.shape, c.shape)
    if c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2] < 0:
        ic(hits)
        ic(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])
    r = np.sqrt(np.abs(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2]))/200
    c = np.concatenate([-c[:, 0]/2, -c[:, 1]/2], axis=-1)
    return r[0], c[0]


def get_approximate_radii(tracks, good_hits, n_layers):
    radii = np.zeros(tracks.shape[0])
    centers = np.zeros((tracks.shape[0], 2))
    for i in range(len(tracks)):
        hits = tracks[i, :15].reshape(5, 3)
        hits = hits[good_hits[i]]
        radii[i], centers[i] = get_radius(hits)

    return radii, centers


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
            use_center=True,
            use_predicted_pz=False,
            use_momentum=False,
            use_transverse_momentum=False,
            use_parallel_momentum=False,
            use_physics_pred=False,
            use_energy=False,
            use_n_hits=False,
            use_n_pixels=False,
            use_trigger=True,
            use_nontrigger=True,
            use_filter=False,
            filter_n_hits=5,
            rescale_by_percentile=-1,
            percentiles=PERCENTILES
            ):
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

        if 0 <= rescale_by_percentile <= 100:
            self.rescale_factor = percentiles[rescale_by_percentile]
        else:
            self.rescale_factor = 1
       
        self.load_complete_graph = load_complete_graph
        self.add_geo_features = add_geo_features
        self.use_radius = use_radius
        self.use_center = use_center
        self.use_predicted_pz = use_predicted_pz
        self.use_momentum = use_momentum
        self.use_transverse_momentum = use_transverse_momentum
        self.use_parallel_momentum = use_parallel_momentum
        self.use_energy = use_energy
        self.use_n_hits = use_n_hits
        self.use_n_pixels = use_n_pixels
        self.use_physics_pred = use_physics_pred
        self.use_filter = use_filter
        self.filter_n_hits = filter_n_hits
        self.z = 0
        self.phi = 0
        self.noise_std = 0

    def __getitem__(self, file_index):
        event_info = load_graph(self.filenames[file_index], self.load_complete_graph, phi=self.phi, z=self.z, noise_std=self.noise_std)
        track_vector = event_info.track_vector / self.rescale_factor
        if self.use_filter:
            hits = track_vector[:, :15].reshape(track_vector.shape[0], 5, 3)
            good_hits = np.any(hits != 0, axis=-1)
            n_hits = np.sum(good_hits, axis=-1)
            if track_vector.shape[0] > 0:
                p_t = np.sqrt(event_info.momentums[:, 0]**2 + event_info.momentums[:, 1]**2)
                mask = np.logical_and(n_hits == self.filter_n_hits, p_t > 0.2)
            else:
                mask = n_hits == self.filter_n_hits
            track_vector = track_vector[mask]
        else:
            mask = np.ones(track_vector.shape[0], dtype=bool)

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
            r = event_info.r[mask] / self.rescale_factor
            track_vector = np.concatenate([track_vector, r], axis=-1)
        elif self.use_radius:
            track_vector = np.concatenate([track_vector, np.zeros((0, 1))], axis=-1)


        if self.use_center and track_vector.shape[0] != 0:
            c = event_info.center[mask] / self.rescale_factor
            track_vector = np.concatenate([track_vector, c], axis=-1)
        elif self.use_center:
            track_vector = np.concatenate([track_vector, np.zeros((0, 2))], axis=-1)

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
            pred_pz = get_predicted_pz(first_hit, last_hit, event_info.r.reshape(-1) / self.rescale_factor)
            track_vector = np.concatenate([track_vector, np.expand_dims(pred_pz, -1)[mask]], axis=-1)
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

        if self.use_n_pixels:
            track_vector = np.concatenate([track_vector, event_info.n_pixels[mask]], axis=-1)

        if self.use_n_hits:
            track_vector = np.concatenate([track_vector, event_info.n_hits[mask]], axis=-1)

        if track_vector.shape[0] != 0:
            origin_vertices = event_info.origin_vertices[mask]
        else:
            origin_vertices = np.zeros((0, 3))


        return BatchInfo(
                track_vector=torch.from_numpy(track_vector).to(torch.float),
                n_tracks=n_tracks,
                trigger=torch.from_numpy(event_info.trigger),
                is_trigger_track=torch.from_numpy(event_info.is_trigger_track),
                momentums=torch.from_numpy(momentums),
                energies=torch.from_numpy(energies),
                ip=torch.from_numpy(event_info.ip),
                origin_vertices=torch.from_numpy(origin_vertices),
                ptypes=torch.from_numpy(event_info.ptypes[mask]),
            )

    def __len__(self):
        return len(self.filenames)


def get_datasets(n_train, n_valid, n_test, 
        trigger_input_dir=None, 
        nontrigger_input_dir=None,
        load_complete_graph=False, 
        add_geo_features=True,
        use_radius=True,
        use_center=True,
        use_predicted_pz=False,
        use_momentum=False,
        use_transverse_momentum=False,
        use_parallel_momentum=False,
        use_physics_pred=False,
        use_energy=False,
        use_n_pixels=False,
        use_n_hits=False,
        use_trigger=True,
        use_nontrigger=True,
        use_filter=False,
        filter_n_hits=5,
        rescale_by_percentile=-1,
        percentiles=PERCENTILES):
    data = TrackDataset(trigger_input_dir=trigger_input_dir,
                        nontrigger_input_dir=nontrigger_input_dir,
                        n_trigger_samples=n_train+n_valid+n_test,
                        n_nontrigger_samples=n_train+n_valid+n_test,
                        load_complete_graph=load_complete_graph,
                        add_geo_features=add_geo_features,
                        use_radius=use_radius,
                        use_center=use_center,
                        use_predicted_pz=use_predicted_pz,
                        use_momentum=use_momentum,
                        use_transverse_momentum=use_transverse_momentum,
                        use_parallel_momentum=use_parallel_momentum,
                        use_physics_pred=use_physics_pred,
                        use_energy=use_energy,
                        use_trigger=use_trigger,
                        use_nontrigger=use_nontrigger,
                        use_filter=use_filter,
                        filter_n_hits=filter_n_hits,
                        use_n_pixels=use_n_pixels,
                        use_n_hits=use_n_hits)

    total = use_trigger + use_nontrigger
    train_data, valid_data, test_data = random_split(data, [total*n_train, total*n_valid, total*n_test])

    return train_data, valid_data, test_data
