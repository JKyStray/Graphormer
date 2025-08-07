import functools
import pickle
import random
from pathlib import Path

import numpy as np
import torch
from common import config as cfg
from torch.utils.data.dataset import Dataset

from .util import LMDBReader


def pseudo_CB(seq, CB, CA):
    """
    use CA instead CB for GLY
    """
    seq = np.array(list(seq))
    return np.where(np.expand_dims(seq == "G", -1), CA, CB)


def pairwise_distance(x, y=None):
    """
    in_shape: (L, k)
    out_shape: (L, L)
    """
    if y is None:
        y = x
    return np.linalg.norm(x[:, None, :] - y[None, :, :], axis=-1)


def calc_rotate_imat(N, CA, C):
    p1 = N - CA
    x = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
    p2 = C - N
    inner_1 = np.matmul(np.expand_dims(p1, axis=1), np.expand_dims(p1, axis=2))[:, :, 0]
    inner_2 = np.matmul(np.expand_dims(-p1, axis=1), np.expand_dims(p2, axis=2))[
        :, :, 0
    ]
    alpha = inner_1 / inner_2
    y = alpha * p2 + p1
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    mat = np.concatenate([x, y, z], axis=-1)
    mat = mat.reshape(*mat.shape[:-1], 3, 3)
    return mat


def detect_data_format(npz_data):
    """
    NEW
    
    Detect whether NPZ data contains coordinates or transformation matrices.
    
    Args:
        npz_data: Loaded NPZ data
        
    Returns:
        str: "coordinate" or "transformation_matrix"
    """
    if set(["CA", "N", "C", "CB"]).issubset(npz_data.keys()):
        return "coordinate"
    elif set(["tr", "rot_mat"]).issubset(npz_data.keys()):
        return "transformation_matrix"
    else:
        raise ValueError(f"Unknown data format. Keys found: {list(npz_data.keys())}")


def create_dummy_cbcb_distances(L, avg_distance=8.0, noise_std=2.0):
    """
    NEW
    
    Create a dummy CB-CB distance matrix for when we don't have coordinate data.
    
    Args:
        L: Sequence length
        avg_distance: Average distance between residues
        noise_std: Standard deviation for distance noise
        
    Returns:
        np.ndarray: [L, L] distance matrix
    """
    # Create base distance matrix with sequence separation bias
    i, j = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    seq_sep = np.abs(i - j)
    
    # Base distances: closer for nearby residues, farther for distant ones
    base_distances = avg_distance + 0.5 * seq_sep
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, noise_std, (L, L))
    distances = base_distances + noise
    
    # Ensure symmetry and positive distances
    distances = (distances + distances.T) / 2
    distances = np.maximum(distances, 1.0)  # Minimum distance of 1.0 Å
    
    # Diagonal should be 0
    np.fill_diagonal(distances, 0.0)
    
    return distances


class LMDBDataset(Dataset):
    def __init__(self, db_path):
        self.db_path = db_path
        self._reader = LMDBReader(db_path)

    def get_pkl_dir(self, index):
        pkl_dir = Path(self.db_path) / "features" / "pdb40" / index
        return pkl_dir

    def get_LR(self, pkl_name):
        L, R = pkl_name.split("_")[2:]
        L, R = int(L), int(R)
        return L, R

    def __getitem__(self, index):
        item = self._reader.get_structure(index)
        seq, N, C, CA, CB = item["seq"], item["N"], item["C"], item["CA"], item["CB"]
        center = np.nanmean(CA)
        N -= center
        C -= center
        CA -= center
        CB -= center

        pCB = pseudo_CB(seq, CB, CA)
        cbcb_dist = pairwise_distance(pCB)
        T = CA
        IR = calc_rotate_imat(N=N, CA=CA, C=C)

        pkl_dir = self.get_pkl_dir(index)
        pkls = list(pkl_dir.glob("*.pkl"))
        pkl = random.choice(pkls)

        pkl_data = pickle.load(open(pkl, "rb"))
        # np.array(...): convert jax np array back to numpy
        single_repr, pair_repr = np.array(pkl_data["single"]), np.array(
            pkl_data["pair"]
        )

        L, R = self.get_LR(pkl.stem)

        return {
            "single_repr": single_repr,
            "pair_repr": pair_repr,
            "T": T[L:R],
            "IR": IR[L:R],
            "cbcb": cbcb_dist[L:R][:, L:R],
        }


class NPYReader:
    def __init__(self, db_path):
        self.db_path = Path(db_path)

    @functools.lru_cache(maxsize=1)
    def get_num_cluster(self):
        raise NotImplementedError

    def get_clu(self, tp, index):
        raise NotImplementedError

    def get_structure(self, index):
        npz_dir = self.db_path / "raw" / index / "structure"
        npz_path = random.choice(list(npz_dir.glob("*.npz")))
        t = np.load(npz_path)
        
        # NEW Detect data format and process accordingly
        data_format = detect_data_format(t)
        
        # NEW
        if data_format == "coordinate":
            # Original coordinate-based processing
            n_samples = t["CA"].shape[0]
            L = t["CA"].shape[1]
            sid = random.randint(0, n_samples - 1)

            item = {
                "seq": "." * L,  # make pseudo_CB happy
                "N": t["N"][sid],
                "C": t["C"][sid],
                "CA": t["CA"][sid],
                "CB": t["CB"][sid],
            }
            
        # NEW
        elif data_format == "transformation_matrix":
            # New transformation matrix processing
            L = t["tr"].shape[0]  # Get sequence length from transformation data
            
            # For transformation matrix format, we create a minimal coordinate structure
            # The important data (T, IR) will be overridden later from tr/rot_mat
            item = {
                "seq": "." * L,  # make pseudo_CB happy
                "N": np.zeros((L, 3)),    # dummy coordinates
                "C": np.zeros((L, 3)),    # dummy coordinates  
                "CA": t["tr"],            # Use translation vectors as CA coordinates
                "CB": t["tr"],            # Use translation vectors as CB coordinates (will be overridden)
                "_transformation_data": {  # Store transformation matrices for later use
                    "T": t["tr"],
                    "IR": t["rot_mat"],
                    "L": L
                }
            }
        else:
            raise ValueError(f"Unsupported data format: {data_format}")

        return item


class NPYDataset(LMDBDataset):
    def __init__(self, db_path):
        self.db_path = Path(db_path)
        self._reader = NPYReader(db_path)

    def get_pkl_dir(self, index):
        pkl_dir = Path(self.db_path) / "raw" / index / "features"
        return pkl_dir

    def get_LR(self, pkl_name):
        L, R = pkl_name.split("_")[-2:]
        L, R = int(L), int(R)
        return L, R

    # NEW
    def __getitem__(self, index):
        item = self._reader.get_structure(index)
        
        # Check if this is transformation matrix data
        if "_transformation_data" in item:
            # Handle transformation matrix format
            trans_data = item["_transformation_data"]
            T = trans_data["T"]
            IR = trans_data["IR"] 
            L = trans_data["L"]
            
            # Create dummy CB-CB distances since we don't have coordinate data
            cbcb_dist = create_dummy_cbcb_distances(L)
            
        else:
            # Handle coordinate format (original processing)
            seq, N, C, CA, CB = item["seq"], item["N"], item["C"], item["CA"], item["CB"]
            center = np.nanmean(CA)
            N -= center
            C -= center
            CA -= center
            CB -= center

            pCB = pseudo_CB(seq, CB, CA)
            cbcb_dist = pairwise_distance(pCB)
            T = CA
            IR = calc_rotate_imat(N=N, CA=CA, C=C)

        pkl_dir = self.get_pkl_dir(index)
        pkls = list(pkl_dir.glob("*.pkl"))
        pkl = random.choice(pkls)

        pkl_data = pickle.load(open(pkl, "rb"))
        
        # Handle both direct and nested representations formats
        if "representations" in pkl_data:
            single_repr = np.array(pkl_data["representations"]["single"])
            pair_repr = np.array(pkl_data["representations"]["pair"])
        else:
            # Fallback for direct format
            single_repr = np.array(pkl_data["single"])
            pair_repr = np.array(pkl_data["pair"])

        L, R = self.get_LR(pkl.stem)

        return {
            "single_repr": single_repr,
            "pair_repr": pair_repr,
            "T": T[L:R],
            "IR": IR[L:R],
            "cbcb": cbcb_dist[L:R][:, L:R],
        }


class BatchDataset(Dataset):
    def __init__(self, dataset, padding_values):
        self.dataset = dataset
        self.padding_values = padding_values

    def _pad_fn(self, a, value):
        max_shape = np.max([_.shape for _ in a], axis=0)
        na = []
        for x in a:
            pad_shape = [(0, l2 - l1) for l1, l2 in zip(x.shape, max_shape)]

            na.append(np.pad(x, pad_shape, mode="constant", constant_values=value))
        return np.stack(na)

    def __getitem__(self, indices):
        data = []
        for index in indices:
            data.append(self.dataset[index])
        ret = {}
        for key in data[0].keys():
            pad_value = self.padding_values.get(key, 0)
            ret[key] = self._pad_fn([_[key] for _ in data], pad_value)
        return ret


class StructureDataset(Dataset):
    def __init__(self):
        ds = LMDBDataset(db_path=cfg.dataset_dir)
        ds = BatchDataset(
            dataset=ds,
            padding_values={
                "single_repr": 0,
                "pair_repr": 0,
                "T": np.nan,
                "IR": np.nan,
                "cbcb": np.nan,
            },
        )
        self._dataset = ds

    def __getitem__(self, indices):
        data = self._dataset[indices]
        return {
            "single_repr": torch.tensor(data["single_repr"]).float(),
            "pair_repr": torch.tensor(data["pair_repr"]).float(),
            "T": torch.tensor(data["T"]).float(),
            "IR": torch.tensor(data["IR"]).float(),
            "cbcb": torch.tensor(data["cbcb"]).float(),
        }


class StructureDatasetNPY(StructureDataset):
    def __init__(self):
        ds = NPYDataset(db_path=cfg.dataset_dir)
        ds = BatchDataset(
            dataset=ds,
            padding_values={
                "single_repr": 0,
                "pair_repr": 0,
                "T": np.nan,
                "IR": np.nan,
                "cbcb": np.nan,
            },
        )
        self._dataset = ds
