# Copyright 2022 The Procter & Gamble Company
#
# Licenced under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the original License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

""" Utility functions being reused in different modules.
"""
import numpy as np
import sklearn.neighbors


def ts2windows(arr, seqlen, step=1, stride=1, padding='after'):
    assert padding in [ 'before', 'after', None, ]
    arrlen = len(arr) # original length of the array.
    if padding is None:
        n = seqlen // 2
        arr = arr.flatten()
        windows = [ arr[i-n:i-n+seqlen:step] for i in range(n, arrlen-n, stride) ]
    else:
        pad_width = (seqlen - 1, 0) if padding == 'before' else (0, seqlen - 1)
        arr = np.pad(arr.flatten(), pad_width, mode="constant", constant_values=0)
        windows = [ arr[i:i+seqlen:step] for i in range(0, arrlen, stride) ]

    windows = np.array(windows, dtype="float32")
    return windows


def l1norm(x1, x2):
    ret = np.abs(x2 - x1)
    if len(ret.shape) > 1:
        if ret.shape[1] > 1:
            ret = np.sum(ret, axis=1).reshape((-1, 1))
    return ret


def l2norm(x1, x2):
    return np.sqrt(np.sum((x2 - x1)**2))


def nearest_neighbor(mat):
    """
    Using Ball Tree nearest neighbours algorithm as it scales well and is deterministic (no random changes in the indices).
    Sklearn implementation of nearest neighbors returns some windows as nearest neighbors of themselves.
    This function ensures that all the neighbors are distinct from the input points.
    """
    n_pts = mat.shape[0]
    leafsz = max([ 3, n_pts // (1000 * mat.shape[1]), ])
    neighbors = sklearn.neighbors.BallTree(mat, leaf_size=leafsz, metric="euclidean")
    dist_mat, idx_mat = neighbors.query(mat, k=2, dualtree=True)
    self_nn = np.arange(n_pts) == idx_mat[:,0]
    true_idx = np.where(self_nn, idx_mat[:,1], idx_mat[:,0]).reshape((-1, 1))
    true_dist = np.where(self_nn, dist_mat[:,1], dist_mat[:,0]).reshape((-1, 1))
    return true_dist, true_idx

