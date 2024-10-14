import numpy as np
from numpy.fft import fft, ifft
import cupy as cp
import copy
from tqdm import tqdm



def simple_fast(data, query, window_size):
    """Get the matrix profile and profile index of data and a query."""
    if data.shape[1] > data.shape[0]:
        data = data.T
        query = query.T
    if query.shape[1] != data.shape[1]:
        raise ValueError(
            f"incompatible query dimension: {query.shape} against {data.shape}"
        )

    # only used for self-join
    exclusion_zone = (
        round(window_size / 2)
        if data.shape == query.shape and np.allclose(data, query)
        else 0
    )

    n, dim = query.shape

    matrix_profile_length = data.shape[0] - window_size + 1
    matrix_profile = np.zeros(matrix_profile_length)
    profile_index = np.zeros(matrix_profile_length)

    # compute the first dot-product for the data and query
    X, sumx2 = mass_pre(data, window_size)
    _, z0, _ = mass(X, query[:window_size], data.shape[0], window_size, dim, sumx2)

    # compute the first distance profile
    X, sumx2 = mass_pre(query, window_size)
    distance_profile, z, sumy2 = mass(X, data[:window_size], n, window_size, dim, sumx2)
    dropval = data[0]

    # only done on self joins
    distance_profile[:exclusion_zone] = np.Inf

    # compute the first distance profile
    idx = np.argmin(distance_profile)
    profile_index[0] = idx
    matrix_profile[0] = distance_profile[idx]

    # compute the rest of the matrix profile
    nz, _ = z.shape
    for i in range(1, matrix_profile_length):
        subsequence = data[i : i + window_size]
        sumy2 = sumy2 - dropval ** 2 + subsequence[-1] ** 2
        z[1:nz] = (
            z[: nz - 1]
            + subsequence[-1] * query[window_size : window_size + nz - 1]
            - dropval * query[: nz - 1]
        )

        z[0] = z0[i]
        dropval = subsequence[0]

        distance_profile = (sumx2 - 2 * z + sumy2).sum(axis=1)

        if exclusion_zone > 0:
            start = max(0, i - exclusion_zone)
            end = min(matrix_profile_length, i + exclusion_zone + 1)
            distance_profile[start:end] = np.Inf

        idx = np.argmin(distance_profile)
        profile_index[i] = idx
        matrix_profile[i] = distance_profile[idx]

    return matrix_profile, profile_index.astype(int)


def mass_pre(x, m):
    """m is the window size."""
    n, dim = x.shape
    x_mat = np.zeros((2 * n, dim))
    x_mat[:n] = x
    X = fft(x_mat, axis=0)
    cum_sumx2 = (x ** 2).cumsum(axis=0)
    sumx2 = cum_sumx2[m - 1 : n] - np.append(
        np.zeros((1, dim)), cum_sumx2[: n - m], axis=0
    )
    return X, sumx2


def mass(X, y, n, m, dim, sumx2):
    """Calculate the distance profile using the MASS algorithm.

    X: the fft data
    y: the query data
    n: the number of rows in the query
    m: the sliding window size
    dim: the number of dimensions
    sumx2: the precomputed sum of squares

    returns (dist, z, sumy2)
    where
        dist: the distance profile
        z: the last product
        sumy2: the sum of squared query values
    """

    # computing dot product in O(n log n) time
    y_mat = np.zeros((2 * n, dim))
    y_mat[:m] = y[::-1]
    Y = fft(y_mat, axis=0)
    Z = X * Y
    z = np.real(ifft(Z, axis=0)[m - 1 : n])

    # compute y stats O(n)
    sumy2 = (y ** 2).sum(axis=0)

    # compute distances O(n)
    dist = (sumx2 - 2 * z + sumy2).sum(axis=1)
    return dist, z, sumy2

def batch_simple_fast_gpu(data, query, window_size):
    """Get the matrix profile and profile index of data and a query."""
    if data.shape[1] > data.shape[0]:
        data = data.T
        query = query.T
    if query.shape[2] != data.shape[1]:
        raise ValueError(
            f"incompatible query dimension: {query.shape} against {data.shape}"
        )

    # only used for self-join
    exclusion_zone = np.array([round(window_size / 2) if data.shape == que.shape and np.allclose(data, que) else 0 for que in query])
    
    query = cp.asarray(query)
    data = cp.asarray(data)

    n_sample, n, dim = query.shape

    matrix_profile_length = data.shape[0] - window_size + 1
    matrix_profile = cp.zeros((n_sample, matrix_profile_length))
    profile_index = cp.zeros((n_sample, matrix_profile_length))

    # compute the first dot-product for the data and query
    X, sumx2 = mass_pre_single(data, window_size) # change this
    _, z0, _ = mass_single_multi(X, query[:, :window_size, :], data.shape[0], window_size, dim, sumx2, n_sample) # change this

    # compute the first distance profile
    X, sumx2 = mass_pre_multi(query, window_size)
    distance_profile, z, sumy2 = mass_multi_single(X, data[:window_size], n, window_size, dim, sumx2, n_sample)
    dropval = data[0]

    # only done on self joins
    for i in range(n_sample):
        distance_profile[i, :exclusion_zone[i]] = cp.Inf

    idx = cp.argmin(distance_profile, axis = 1)
    profile_index[:, 0] = idx 
    for i in range(matrix_profile.shape[0]):
        matrix_profile[i, 0] = distance_profile[i, idx[i]] 

    _,nz, _ = z.shape
    sliced_query_1 = query[:, window_size : window_size + nz - 1, :]
    sliced_query_2 = query[:, : nz - 1, :]
    sumy23 = copy.deepcopy(sumy2[0])
    sumy23 = cp.asarray(sumy23)

    for i in tqdm(range(1, matrix_profile_length)):
        subsequence = data[i : i + window_size]
        sumy23 = sumy23 - dropval ** 2 + subsequence[-1] ** 2
        z[:, 1:nz, :] = (
            z[:, : nz - 1, :]
            + subsequence[-1] * sliced_query_1
            - dropval * sliced_query_2
        )
        z[:,0,:] = z0[:,i,:]
        dropval = subsequence[0]
        distance_profile = (sumx2 - 2 * z + cp.tile(sumy23, (n_sample, 1))[:, cp.newaxis, :]).sum(axis=2)
        
        for ex in range(n_sample):
            if exclusion_zone[ex] > 0:
                start = max(0, i - exclusion_zone[ex])
                end = min(matrix_profile_length, i + exclusion_zone[ex] + 1)
                distance_profile[ex, start:end] = cp.Inf
                
        idx = cp.argmin(distance_profile, axis = 1) 
        profile_index[:, i] = idx
        matrix_profile[:, i] = distance_profile[cp.arange(matrix_profile.shape[0]), idx]
    return matrix_profile, profile_index.astype(int)



def mass_pre_single(x, m):
    """m is the window size."""
    n, dim = x.shape
    x_mat = cp.zeros((2 * n, dim))
    x_mat[:n] = x
    # print(x_mat.shape)
    X = cp.fft.fft(x_mat, axis=0)
    cum_sumx2 = (x ** 2).cumsum(axis=0)
    sumx2 = cum_sumx2[m - 1 : n] - cp.append(
        cp.zeros((1, dim)), cum_sumx2[: n - m], axis=0
    )
    # print("mass pre ", x_mat.shape, sumx2.shape)
    return X, sumx2

def mass_pre_multi(x, m):
    """m is the window size."""
    n_sample, n, dim = x.shape
    x_mat = cp.zeros((n_sample, 2 * n, dim))
    x_mat[:, :n, :] = x
    X = cp.fft.fft(x_mat, axis=1)
    cum_sumx2 = (x ** 2).cumsum(axis=1)
    sumx2 = cum_sumx2[:, m - 1 : n, :] - cp.append(
        cp.zeros((n_sample, 1, dim)), cum_sumx2[:, : n - m, :], axis=1
    )    
    return X, sumx2


def mass_multi_single(X, y, n, m, dim, sumx2, n_sample):
    y = cp.tile(y, (n_sample, 1, 1))
    y_mat = cp.zeros((n_sample, 2 * n, dim))
    y_mat[:,:m, :] = y[:,::-1, :]
    Y = cp.fft.fft(y_mat, axis=1)
    Z = X * Y
    z = cp.real(cp.fft.ifft(Z, axis=1)[:, m - 1 : n, :])

    # compute y stats O(n)
    sumy2 = (y ** 2).sum(axis=1)
    # compute distances O(n)
    dist = (sumx2 - 2 * z + sumy2[:, cp.newaxis, :]).sum(axis=2)
    return dist, z, sumy2


def mass_single_multi(X, y, n, m, dim, sumx2, n_sample):
    y_mat = cp.zeros((n_sample, 2 * n, dim))
    y_mat[:,:m, :] = y[:,::-1, :]
    Y = cp.fft.fft(y_mat, axis=1)
    X = cp.tile(X, (n_sample, 1, 1))
    sumx2 = cp.tile(sumx2, (n_sample, 1, 1))
    Z = X * Y
    z = np.real(cp.fft.ifft(Z, axis=1)[:, m - 1 : n, :])

    sumy2 = (y ** 2).sum(axis=1)

    # compute distances O(n)
    dist = (sumx2 - 2 * z + sumy2[:, cp.newaxis, :]).sum(axis=2)
    return dist, z, sumy2


def multi_simple_fast(one_feat, multi_feat, window_size=30):
    mprs = []
    for chunk_feat in tqdm(multi_feat):
        mpr = simple_fast(one_feat, chunk_feat, 30)[0]
        mprs.append(mpr)
    return np.vstack(mprs)