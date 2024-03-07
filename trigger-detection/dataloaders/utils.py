import numpy as np
from numpy.linalg import inv
def get_track_endpoints(hits, good_layers):
    # Assumption: all tracks have at least 1 hit
    # If it has one hit, first_hit == last_hit for that track
    # hits shape: (n_tracks, 5, 3)
    # good_layers shape: (n_tracks, 5)
    min_indices = good_layers * np.arange(5) + (1 - good_layers) * np.arange(5, 10)
    indices = np.expand_dims(np.argmin(min_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    first_hits = np.take_along_axis(hits, indices, axis=-2)
    max_indices = good_layers * np.arange(5, 10) + (1 - good_layers) * np.arange(5)
    indices = np.expand_dims(np.argmax(max_indices, axis=-1), -1)
    indices = np.expand_dims(indices, axis=-2)
    last_hits = np.take_along_axis(hits, indices, axis=-2)
    return first_hits.squeeze(1), last_hits.squeeze(1)

def get_predicted_pz(track_hits, good_layers, radius):
    hits = track_hits.reshape(-1, 5, 3)
    first_hit, last_hit = get_track_endpoints(hits, good_layers)
    dz = (last_hit[:, -1] - first_hit[:, -1])/100
    chord2 = ((last_hit[:, 0] - first_hit[:, 0]) ** 2 + (last_hit[:, 1] - first_hit[:, 1]) ** 2) / 10000
    r2 = 2*radius**2
    with np.errstate(invalid='ignore'):
        dtheta = np.arccos((r2 - chord2) / (r2 + (r2 == 0)))
    dtheta += (dtheta == 0) * 1e-6
    return np.nan_to_num(dz / dtheta)

def matmul_3D(A, B):
    return np.einsum('lij,ljk->lik', A, B)


def get_approximate_radii(track_hits, good_layers, n_layers):
    x_indices = [3*j for j in range(5)]
    y_indices = [3*j+1 for j in range(5)]
    r = np.zeros(track_hits.shape[0])
    centers = np.zeros((track_hits.shape[0], 2))
    for n_layer in range(3, 5 + 1):
        complete_tracks = track_hits[n_layers == n_layer]
        hit_indices = good_layers[n_layers == n_layer]
        if complete_tracks.shape[0] == 0:
            continue

        A = np.ones((complete_tracks.shape[0], n_layer, 3))
        x_values = complete_tracks[:, x_indices]
        x_values = x_values[hit_indices].reshape(complete_tracks.shape[0], n_layer)

        y_values = complete_tracks[:, y_indices]
        y_values = y_values[hit_indices].reshape(complete_tracks.shape[0], n_layer)
        A[:, :, 0] = x_values
        A[:, :, 1] = y_values

        y = - x_values**2 - y_values**2
        y = y.reshape((y.shape[0], y.shape[1], 1))
        AT = np.transpose(A, axes=(0, 2, 1))
        c = matmul_3D(matmul_3D(inv(matmul_3D(AT, A)), AT), y)[..., 0]
        r[n_layers == n_layer] = np.sqrt(c[:, 0]**2 + c[:, 1]**2 - 4*c[:, 2])/200
        centers[n_layers == n_layer] = np.stack([-c[:, 0]/2, -c[:, 1]/2], axis=-1)

    #test = get_approximate_radius(track_hits, n_layers == 5)
    #assert np.allclose(test, r[n_layers == 5])

    return r, centers

def get_length(start, end):
    return np.sqrt(np.sum((start - end)**2, axis=1))


def calculate_geometric_features(track_hits):
    # 4 edge length + 1 total length, 1 angle + 4 delta angle, hits center , total 13
    geo_features = np.zeros((track_hits.shape[0], 13))
    if track_hits.shape[0] == 0:
        return geo_features

    phi  = np.zeros((track_hits.shape[0], 5))
    geo_features[:, 5] = np.arctan2(track_hits[:, 1], track_hits[:, 0])
    for i in range(4):
        geo_features[:, i] = get_length(
                track_hits[:, (3*i+3):(3*i+6)], 
                track_hits[:, (3*i):(3*i+3)]
                )
    for i in range(5):
        phi[:, i] = np.arctan2(
                track_hits[:, (3*i)+1], 
                track_hits[:, (3*i)]
                )
    geo_features[:, 5] = get_length(
            track_hits[:, 12:15], 
            track_hits[:, 0:3]
            )
    geo_features[:, 6:10] = np.diff(phi)
    geo_features[:, 10:13] = np.mean(
        track_hits.reshape((track_hits.shape[0], 5, 3)), axis=(0, 1)
    )

    return geo_features
