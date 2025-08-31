from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import mdtraj as md
import pandas as pd


@dataclass
class ProjectionSettings:
    """
    Settings for building loop-local features and projecting to 2D.

    Attributes:
        exclude_neighbors: Exclude neighbors within N residues along the sequence when
            computing pairwise distances (to reduce trivial short-range contacts).
        effective_distance: Scale for contact transform; distances are divided by this
            value before applying exp(-d/scale) and clamped to 1.0.
        use_pca: If True, compute a 2D PCA on the MD features and apply to inference.
            If False, use whitening via inverse sqrt covariance; if feature dim < 2,
            fallback to PCA.
    """

    exclude_neighbors: int = 2
    effective_distance: float = 0.8
    use_pca: bool = True


@dataclass
class CoverageSettings:
    """
    Settings for density estimation and coverage evaluation in 2D.

    Attributes:
        n_resample: Number of resamples with Gaussian noise for stable binning.
        sigma_resample: Standard deviation of Gaussian noise added to samples.
        num_bins: Number of bins per axis for 2D histogram.
        energy_cutoff: Low-energy region cutoff relative to minimum free energy (kcal/mol).
        temperature_K: Temperature for converting probabilities to free energy (Kelvin).
        padding: Padding factor to extend histogram edges beyond min/max.
        random_seed: Seed for reproducibility.
    """

    n_resample: int = 50_000
    sigma_resample: float = 0.15
    num_bins: int = 40
    energy_cutoff: float = 5.0
    temperature_K: float = 300.0
    padding: float = 0.5
    random_seed: int | None = 42


# Boltzmann constant in kcal/mol/K
K_BOLTZMANN: float = 0.0019872041


def list_pdb_files(directory: str | Path, max_files: int | None = None) -> List[Path]:
    directory = Path(directory).expanduser().resolve()
    pdbs = sorted([p for p in directory.glob("*.pdb") if p.is_file()])
    if max_files is not None:
        pdbs = pdbs[:max_files]
    return pdbs


def load_trajectory_from_pdb_dir(directory: str | Path, max_files: int | None = None) -> md.Trajectory:
    """
    Load and concatenate PDB frames from a directory into a single trajectory.
    """
    pdb_files = list_pdb_files(directory, max_files=max_files)
    if len(pdb_files) == 0:
        raise FileNotFoundError(f"No PDB files found in {directory}")
    trajs = [md.load(str(p)) for p in pdb_files]
    return md.join(trajs)


def get_ca_atom_indices_by_residue_order(topology: md.Topology) -> List[int]:
    """
    Return CA atom indices ordered by residue iteration order (0..N-1).
    This avoids relying on PDB resid numbering.
    """
    ca_indices: List[int] = []
    for residue in topology.residues:
        ca_idx = None
        for atom in residue.atoms:
            if atom.name == "CA":
                ca_idx = atom.index
                break
        if ca_idx is None:
            raise ValueError(f"Residue {residue} lacks a CA atom; cannot proceed.")
        ca_indices.append(ca_idx)
    return ca_indices


def residue_mask_to_ca_indices(mask: np.ndarray, ca_indices_by_residue: List[int]) -> np.ndarray:
    if len(mask) != len(ca_indices_by_residue):
        raise ValueError(
            f"Mask length {len(mask)} does not match number of residues {len(ca_indices_by_residue)}"
        )
    return np.array([ca_indices_by_residue[i] for i in range(len(mask)) if bool(mask[i])], dtype=int)


def superpose_on_framework(reference: md.Trajectory, target: md.Trajectory, framework_ca_indices: np.ndarray) -> md.Trajectory:
    """
    Return a copy of target superposed onto reference using framework CA atoms.
    """
    aligned = target[:]
    aligned.superpose(reference, atom_indices=framework_ca_indices, ref_atom_indices=framework_ca_indices)
    return aligned


def compute_loop_ca_coordinates(traj: md.Trajectory, loop_ca_indices: np.ndarray) -> np.ndarray:
    """
    Extract CA coordinates for a loop across trajectory frames: shape [n_frames, n_loop_ca, 3].
    Coordinates are in nm.
    """
    return traj.xyz[:, loop_ca_indices, :]


def compute_contact_features_from_ca(
    ca_coordinates: np.ndarray,
    exclude_neighbors: int,
    effective_distance: float,
) -> np.ndarray:
    """
    Compute transformed CA-CA contact features within a loop.

    Steps:
        - Pairwise distances across CA atoms per frame
        - Zero out neighbors within exclude_neighbors along sequence index
        - Transform: x = exp(-d / effective_distance), clamped to 1.0
        - Vectorize upper-triangular entries

    Returns: features with shape [n_frames, n_features]
    """
    n_frames, n_ca, _ = ca_coordinates.shape
    # Pairwise distances: [n_frames, n_ca, n_ca]
    diffs = ca_coordinates[:, :, None, :] - ca_coordinates[:, None, :, :]
    dists = np.linalg.norm(diffs, axis=-1)

    # Neighbor mask by loop sequence positions
    seq_idx = np.arange(n_ca)
    neighbor_mask = (np.abs(seq_idx[:, None] - seq_idx[None, :]) <= exclude_neighbors)
    dists[:, neighbor_mask] = 0.0

    # Transform
    transformed = np.minimum(np.exp(-dists / effective_distance), 1.0)

    # Upper triangular indices (including diagonal for stability)
    iu, ju = np.triu_indices(n_ca)
    features = transformed[:, iu, ju]
    return features


def fit_projection_2d_on_md(features_md: np.ndarray, use_pca: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit a 2D linear projection on MD features (mean + components) and return projected MD.
    If use_pca is True: PCA via SVD; else whitening to decorrelate and scale by inverse sqrt
    eigenvalues. If features have rank < 2, fallback to PCA.
    Returns: (proj_md [N x 2], mean [D], components [D x 2])
    """
    X = features_md.astype(np.float64)
    mean = X.mean(axis=0)
    Xc = X - mean
    # SVD
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    rank = np.sum(S > 1e-12)

    if use_pca or rank < 2:
        components = Vt[:2, :].T  # [D x 2]
        proj_md = Xc @ components
        return proj_md, mean, components

    # Whitening
    cov = (Xc.T @ Xc) / max(1, Xc.shape[0] - 1)
    evals, evecs = np.linalg.eigh(cov)
    # Sort descending
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    # Inverse sqrt for top-2 components; clamp small eigenvalues
    inv_sqrt = np.diag(1.0 / np.sqrt(np.maximum(evals[:2], 1e-8)))
    components = evecs[:, :2] @ inv_sqrt  # [D x 2]
    proj_md = Xc @ components
    return proj_md, mean, components


def project_features_2d(features: np.ndarray, mean: np.ndarray, components: np.ndarray) -> np.ndarray:
    return (features - mean) @ components


def _histogram_edges(x: np.ndarray, num_bins: int, padding: float) -> Tuple[np.ndarray, np.ndarray]:
    x_min, x_max = np.min(x[:, 0]), np.max(x[:, 0])
    y_min, y_max = np.min(x[:, 1]), np.max(x[:, 1])
    dx = (x_max - x_min) / (num_bins + 1)
    dy = (y_max - y_min) / (num_bins + 1)
    x_min -= padding * dx
    x_max += padding * dx
    y_min -= padding * dy
    y_max += padding * dy
    edges_x = np.linspace(x_min, x_max, num_bins + 1)
    edges_y = np.linspace(y_min, y_max, num_bins + 1)
    return edges_x, edges_y


def _resample_with_noise(x: np.ndarray, n_resample: int, sigma: float, seed: int | None) -> np.ndarray:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.shape[0], size=n_resample)
    return x[idx] + sigma * rng.standard_normal((n_resample, x.shape[1]))


def _density_2d(x: np.ndarray, edges_x: np.ndarray, edges_y: np.ndarray) -> np.ndarray:
    H, _, _ = np.histogram2d(x[:, 0], x[:, 1], bins=(edges_x, edges_y), density=True)
    return H


def _density_cutoff_from_energy(density: np.ndarray, kBT: float, energy_cutoff: float) -> float:
    energy_min = -kBT * np.log(np.maximum(density.max(), 1e-32))
    p_cutoff = np.exp(-(energy_min + energy_cutoff) / kBT)
    return float(p_cutoff)


def compute_coverage_metrics(
    reference_proj_2d: np.ndarray,
    sample_proj_2d: np.ndarray,
    settings: CoverageSettings,
) -> Tuple[float, float, float]:
    """
    Compute MAE, RMSE (kcal/mol) and coverage for 2D projections using an MD-emulation style.
    Coverage is the fraction of low-energy reference bins with non-zero sample density.
    """
    kBT = settings.temperature_K * K_BOLTZMANN
    # Resample with noise for stability
    ref_rs = _resample_with_noise(reference_proj_2d, settings.n_resample, settings.sigma_resample, settings.random_seed)
    samp_rs = _resample_with_noise(sample_proj_2d, settings.n_resample, settings.sigma_resample, settings.random_seed)

    # Shared edges from reference
    edges_x, edges_y = _histogram_edges(ref_rs, settings.num_bins, settings.padding)
    rho_ref = _density_2d(ref_rs, edges_x, edges_y)
    rho_samp = _density_2d(samp_rs, edges_x, edges_y)

    # Low-energy mask in reference
    p_cut = _density_cutoff_from_energy(rho_ref, kBT, settings.energy_cutoff)
    low_energy_mask = rho_ref > p_cut

    # Common mask: where low energy in ref and sample is observed
    common_mask = np.logical_and(low_energy_mask, rho_samp > 0)

    # Free energies on masks
    # Clamp densities to avoid -inf
    ref_energy = -kBT * np.log(np.maximum(rho_ref[common_mask], 1e-32))
    samp_energy = -kBT * np.log(np.maximum(rho_samp[common_mask], 1e-32))

    # Minimize wrt global offset
    energy_shift_mae = _optimal_shift_mae(samp_energy, ref_energy)
    mae = float(np.mean(np.abs((samp_energy - ref_energy + energy_shift_mae))))

    shift_rmse = ref_energy.mean() - samp_energy.mean()
    rmse = float(np.sqrt(np.mean((samp_energy - ref_energy + shift_rmse) ** 2)))

    coverage = float(np.count_nonzero(common_mask) / max(1, np.count_nonzero(low_energy_mask)))
    return mae, rmse, coverage


def _optimal_shift_mae(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute the scalar shift that minimizes L1 error between pred + shift and target.
    This is the median of (target - pred).
    """
    delta = target - pred
    return float(np.median(delta))


def evaluate_per_region(
    md_traj: md.Trajectory,
    inf_traj: md.Trajectory,
    cdr_mask: np.ndarray,
    projection_settings: ProjectionSettings | None = None,
    coverage_settings: CoverageSettings | None = None,
    framework_rmsd_threshold_ang: float = 10.0,
) -> pd.DataFrame:
    """
    Evaluate coverage metrics for each CDR region label in cdr_mask (>0).
    If the mask is binary {0,1}, returns a single region named 'CDR'.

    Returns a DataFrame with index=region names and columns ['mae','rmse','coverage']
    including a final 'mean' row with averages.
    """
    projection_settings = projection_settings or ProjectionSettings()
    coverage_settings = coverage_settings or CoverageSettings()

    # Prepare indices
    ca_indices_by_res = get_ca_atom_indices_by_residue_order(md_traj.topology)
    framework_mask = (cdr_mask == 0)
    framework_ca = residue_mask_to_ca_indices(framework_mask, ca_indices_by_res)

    # Superpose both trajectories on MD reference using framework CA atoms
    md_ref = md_traj
    md_aligned = superpose_on_framework(md_ref, md_traj, framework_ca)
    inf_aligned = superpose_on_framework(md_ref, inf_traj, framework_ca)

    # Filter inference frames whose framework RMSD (to md_ref frame 0) exceeds threshold
    # mdtraj uses nm; convert threshold from Å to nm
    threshold_nm = framework_rmsd_threshold_ang / 10.0
    rmsd_nm = md.rmsd(
        inf_aligned,
        md_ref,
        atom_indices=framework_ca,
        ref_atom_indices=framework_ca,
    )
    keep_mask = rmsd_nm <= threshold_nm
    inf_aligned = inf_aligned[keep_mask]

    # Region labels
    labels = np.unique(cdr_mask)
    labels = [int(l) for l in labels if int(l) != 0]
    if len(labels) == 0:
        raise ValueError("Provided CDR mask has no CDR residues (>0)")

    # If binary mask, expose a single 'CDR' region
    label_to_name: Dict[int, str] = {l: f"CDR{l}" for l in labels}
    if set(labels) == {1}:  # binary case
        label_to_name = {1: "CDR"}

    results: Dict[str, Dict[str, float]] = {}

    for label in labels:
        region_name = label_to_name[label]
        region_mask = (cdr_mask == label) if len(labels) > 1 else (cdr_mask > 0)
        loop_ca_idx = residue_mask_to_ca_indices(region_mask, ca_indices_by_res)

        if loop_ca_idx.size < 3:
            # Not enough residues to form meaningful features
            results[region_name] = {"mae": float("nan"), "rmse": float("nan"), "coverage": 0.0}
            continue

        # Build features on MD and inference
        md_ca = compute_loop_ca_coordinates(md_aligned, loop_ca_idx)
        inf_ca = compute_loop_ca_coordinates(inf_aligned, loop_ca_idx)

        md_feat = compute_contact_features_from_ca(
            md_ca,
            exclude_neighbors=projection_settings.exclude_neighbors,
            effective_distance=projection_settings.effective_distance,
        )
        inf_feat = compute_contact_features_from_ca(
            inf_ca,
            exclude_neighbors=projection_settings.exclude_neighbors,
            effective_distance=projection_settings.effective_distance,
        )

        # If no inference frames remain after filtering, set default metrics
        if inf_feat.shape[0] == 0:
            results[region_name] = {"mae": float("nan"), "rmse": float("nan"), "coverage": 0.0}
            continue

        # Fit projection on MD, apply to both
        md_proj2d, mean, components = fit_projection_2d_on_md(md_feat, use_pca=projection_settings.use_pca)
        inf_proj2d = project_features_2d(inf_feat, mean, components)

        # Compute coverage metrics
        mae, rmse, coverage = compute_coverage_metrics(md_proj2d, inf_proj2d, coverage_settings)
        results[region_name] = {"mae": mae, "rmse": rmse, "coverage": coverage}

    # Aggregate
    df = pd.DataFrame(results).T
    df.loc["mean"] = df.mean(numeric_only=True)
    df.index.name = "region"
    return df


def evaluate_from_paths(
    md_dir: str | Path,
    inf_dir: str | Path,
    cdr_mask_path: str | Path,
    max_md_files: int | None = None,
    max_inf_files: int | None = None,
    projection_settings: ProjectionSettings | None = None,
    coverage_settings: CoverageSettings | None = None,
    framework_rmsd_threshold_ang: float = 10.0,
) -> pd.DataFrame:
    """
    Convenience wrapper: load trajectories, load mask, and compute per-region metrics.
    """
    md_traj = load_trajectory_from_pdb_dir(md_dir, max_files=max_md_files)
    inf_traj = load_trajectory_from_pdb_dir(inf_dir, max_files=max_inf_files)
    cdr_mask_path = Path(cdr_mask_path).expanduser().resolve()
    cdr_mask = np.load(str(cdr_mask_path))
    if cdr_mask.ndim != 1:
        raise ValueError(f"CDR mask must be 1D [sequence_length,], got shape {cdr_mask.shape}")

    # If the mask is binary (0/1), attempt to convert it into labeled CDR-specific mask
    unique_vals = set(np.unique(cdr_mask).tolist())
    if unique_vals.issubset({0, 1}):
        try:
            try:
                from .mask_converter import convert_binary_to_labeled_cdr_mask  # type: ignore
            except Exception:
                import sys as _sys
                _HERE = Path(__file__).resolve().parent
                if str(_HERE) not in _sys.path:
                    _sys.path.insert(0, str(_HERE))
                from mask_converter import convert_binary_to_labeled_cdr_mask  # type: ignore

            cdr_mask = convert_binary_to_labeled_cdr_mask(str(cdr_mask_path))
        except Exception:
            # Fall back to treating all CDR as a single region
            pass

    return evaluate_per_region(
        md_traj,
        inf_traj,
        cdr_mask,
        projection_settings,
        coverage_settings,
        framework_rmsd_threshold_ang=framework_rmsd_threshold_ang,
    )

