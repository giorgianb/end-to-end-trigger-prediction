from disjoint_set import DisjointSet
from scipy.stats import mode
import contextlib
import joblib
from tqdm import tqdm
from collections import namedtuple
import numpy as np
import os
from joblib import Parallel, delayed
from icecream import ic
from numpy.linalg import inv

input_dir = 'alltrack_predicted_edge/nontrigger/0/'
output_dir = 'data/alltrack/nontrigger/0/'


def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

# Create named tuple with all the data in the hits file
HitsEventData = namedtuple('HitsEventData', ['hits', 'scaled_hits', 'hits_xyz', 'noise_label', 'layer_id', 'edge_index', 'pid', 'n_hits', 'n_tracks', 'trigger', 'ip', 'psv', 'p_momentum', 'e', 'valid_trigger_flag', 'ParticleTypeID', 'is_complete_trk', 'trigger_track_flag'])


# this is what we need to save in the tracks file:
#        complete_flags = f['is_complete'] 
#        track_vector = f['tracks_info']
#        origin_vertices = f['track_2nd_vertex']
#        momentums = f['momentum'].reshape(-1, 3)
#        pids = f['pid']
#        radius = f['r']
#        is_trigger_track = f['is_trigger_track']
#        ptypes = f['ParticleTypeID']
#        energy = f['energy']
#        trigger = f['trigger_flag']
#        ip = f['ip']
#        n_tracks = f['n_tracks']
#        valid_trigger_flag = f['valid_trigger_flag']

def get_radius(hits):
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

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))



def get_tracks(edge_index):
    # Get connected components
    ds = DisjointSet()
    for i in range(edge_index.shape[1]):
        ds.union(edge_index[0, i], edge_index[1, i])

    return tuple(ds.itersets())

def load_hits_file(filename):
    with np.load(filename, allow_pickle=True) as f:
        hits = f['hits']
        scaled_hits = f['scaled_hits']
        hits_xyz = f['hits_xyz']
        noise_label = f['noise_label']
        layer_id = f['layer_id']
        edge_index = f['edge_index']
        pid = f['pid']
        n_hits = f['n_hits']
        n_tracks = f['n_tracks']
        trigger_flag = f['trigger']
        ip = f['ip']
        psv = f['psv']
        psv = np.stack([p if p is not None else np.zeros(3) for p in psv], axis=0)
        p_momentum = f['p_momentum']
        p_momentum = np.stack([p if p is not None else np.zeros(3) for p in p_momentum], axis=0)
        e = f['e']
        valid_trigger_flag = f['valid_trigger_flag']
        ParticleTypeID = f['ParticleTypeID']
        if ParticleTypeID.dtype is object:
            ic(ParticleTypeID)
        is_complete_trk = f['is_complete_trk']
        trigger_track_flag = f['trigger_track_flag']
        trigger_track_flag = np.array([t if t is not None else False for t in trigger_track_flag]).astype(int)

        # return hits event data. Use keyword arguments to make it easier to read
        return HitsEventData(hits=hits, scaled_hits=scaled_hits, hits_xyz=hits_xyz, noise_label=noise_label, layer_id=layer_id, edge_index=edge_index, pid=pid, n_hits=n_hits, n_tracks=n_tracks, trigger=trigger_flag, ip=ip, psv=psv, p_momentum=p_momentum, e=e, valid_trigger_flag=valid_trigger_flag, ParticleTypeID=ParticleTypeID, is_complete_trk=is_complete_trk, trigger_track_flag=trigger_track_flag)

def construct_track_file(filename, output_filename):
    hits = load_hits_file(filename)
    # tracks has the indices of each hit in each track
    tracks = get_tracks(hits.edge_index.T[(hits.e > 0.5).reshape(-1)].T)

    # a track is 15 numbers, the average location of the hits on each of the 5 layers
    track_vector = np.zeros((len(tracks), 15))
    rs = np.zeros((len(tracks), 1))
    cs = np.zeros((len(tracks), 2))
    complete_flags = np.ones(len(tracks), dtype=bool)
    track_second_vertex = np.zeros((len(tracks), 3))
    momentum = np.zeros((len(tracks), 3))
    pids = np.zeros(len(tracks))
    is_trigger_track = np.zeros(len(tracks), dtype=bool)
    ptypes = np.zeros(len(tracks))
    energy = np.zeros(len(tracks))
    
    for i, track_hits in enumerate(tracks):
        track_hits = list(track_hits)
        hit_layers = hits.layer_id[track_hits].flatten()
        track = np.zeros(15).reshape(5, 3)
        hit_layers[hit_layers == 4] = 3
        hit_layers[hit_layers == 6] = 5
        if len(set(hit_layers)) >= 3:
            r, c = get_radius(hits.hits_xyz[track_hits])
        else:
            r, c = np.zeros((1)), np.zeros((2))


        n_pixels = hits.scaled_hits[track_hits, -1]
        for j in range(5):
            if (hit_layers == j).any():
                track[j] = np.sum(hits.hits_xyz[track_hits][hit_layers == j], axis=0)/np.sum(n_pixels[hit_layers == j])
            else:
                complete_flags[i] = 0
        track_vector[i] = track.reshape(15)
        rs[i] = r
        cs[i] = c
        track_second_vertex[i] = mode(hits.psv[track_hits], axis=0, keepdims=True).mode[0]
        momentum[i] = mode(hits.p_momentum[track_hits], axis=0, keepdims=True).mode[0]
        pids[i] = mode(hits.pid[track_hits], keepdims=False).mode
        is_trigger_track[i] = mode(hits.trigger_track_flag[track_hits], keepdims=False).mode
        ptypes[i] = mode(hits.ParticleTypeID[track_hits], keepdims=False).mode

    hits_center = np.mean(hits.hits_xyz, axis=0)
    geo_features = np.zeros((len(tracks), 13))
    phi  = np.zeros((track_vector.shape[0], 5))
    geo_features[:, 5] = np.arctan2(track_vector[:, 1], track_vector[:, 0])
    for i in range(4):
        geo_features[:, i] = get_length(track_vector[:, (3*i+3):(3*i+6)], track_vector[:, (3*i):(3*i+3)])
    for i in range(5):
        phi[:, i] = np.arctan2(track_vector[:, (3*i)+1], track_vector[:, (3*i)])
    geo_features[:, 5] = get_length(track_vector[:, 12:15], track_vector[:, 0:3])
    geo_features[:, 6:10] = np.diff(phi)
    geo_features[:, 10:13] = hits_center
    track_vector = np.concatenate([track_vector, geo_features], axis=1)


    ip = hits.ip
    n_tracks = len(tracks)
    trigger_flag = hits.trigger

    # Save the data to the tracks file, ensuring that all the fields that we need to save to the tracks file are there:
# this is what we need to save in the tracks file:
#        complete_flags = f['is_complete'] 
#        track_vector = f['tracks_info']
#        origin_vertices = f['track_2nd_vertex']
#        momentums = f['momentum'].reshape(-1, 3)
#        pids = f['pid']
#        radius = f['r']
#        is_trigger_track = f['is_trigger_track']
#        ptypes = f['ParticleTypeID']
#        energy = f['energy']
#        trigger = f['trigger_flag']
#        ip = f['ip']
#        n_tracks = f['n_tracks']
#        valid_trigger_flag = f['valid_trigger_flag']

    np.savez(output_filename, is_complete=complete_flags, tracks_info=track_vector, track_2nd_vertex=track_second_vertex, momentum=momentum, pid=pids, r=rs, c=cs, is_trigger_track=is_trigger_track, ParticleTypeID=ptypes, energy=energy, trigger_flag=trigger_flag, ip=ip, n_tracks=n_tracks, valid_trigger_flag=hits.valid_trigger_flag)


if __name__ == '__main__':
    filenames = sorted(os.listdir(input_dir))
    os.makedirs(output_dir, exist_ok=True)
    def process(filename):
        try:
            input_filename = os.path.join(input_dir, filename)
            output_filename = os.path.join(output_dir, filename)
            construct_track_file(input_filename, output_filename)
        except Exception as e:
            raise
            print(e)

    with tqdm_joblib(tqdm(desc="Conversion", total=len(filenames))) as progress_bar:
        Parallel(n_jobs=16)(delayed(process)(filename) for filename in filenames)
