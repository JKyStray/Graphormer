"""
We want to use this script to evaluate the coverage of the inference results. 

we would like to mimic the MD emulation style from bioemu-benchmarks. 

Our plan is to separately calculate coverage for each CDR. 

Make sure you avoid common pitfalls like not aligning! 

Files and directories:

MD simulation results: ~/Graphormer/distributional_graphormer/protein/dataset/Mars2/raw/A6_2/pdb_structure/
This contain about 10000 pdb files of A6. 

Inference results: ~/Graphormer/distributional_graphormer/protein/output_A6_adaptor_test5_centered_0804/
This contain 250 pdb files of A6 inferred by model. 

CDR mask: ~/Graphormer/distributional_graphormer/protein/evaluation/a6_cdr_mask.npy
This should be [sequence_length, ] where 1 means the residue is in the CDR. You should check 
this file to make sure you understand the structure of this mask. 

Please keep this comment block in the code. 
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

# Support running as a module or directly from this folder
try:  # package-style import
    from .coverage import (
        evaluate_from_paths,
        CoverageSettings,
        ProjectionSettings,
    )
except Exception:  # script-style fallback
    import sys as _sys
    _HERE = Path(__file__).resolve().parent
    if str(_HERE) not in _sys.path:
        _sys.path.insert(0, str(_HERE))
    from coverage import (  # type: ignore
        evaluate_from_paths,
        CoverageSettings,
        ProjectionSettings,
    )


def main() -> None:
    # Configure inputs (edit as needed)
    md_dir = "~/Graphormer/distributional_graphormer/protein/dataset/Mars2/raw/1mi5/pdb_structure/"
    inf_dir = "~/Graphormer/distributional_graphormer/protein/output_1mi5_direct_new_further_0823/"
    cdr_mask_path = "~/Graphormer/distributional_graphormer/protein/evaluation/1mi5_cdr_mask.npy"

    # Settings tuned for ~10k MD vs ~250 inference frames
    proj_settings = ProjectionSettings(exclude_neighbors=2, effective_distance=0.8, use_pca=True)
    cov_settings = CoverageSettings(
        n_resample=50_000,
        sigma_resample=0.15,
        num_bins=40,
        energy_cutoff=5.0,
        temperature_K=300.0,
        padding=0.5,
        random_seed=42,
    )

    df: pd.DataFrame = evaluate_from_paths(
        md_dir=md_dir,
        inf_dir=inf_dir,
        cdr_mask_path=cdr_mask_path,
        max_md_files=None,  # use all; set an int for quick tests
        max_inf_files=None,
        projection_settings=proj_settings,
        coverage_settings=cov_settings,
    )

    # Save & print
    # Write results under this folder regardless of current working directory
    out_dir = (Path(__file__).resolve().parent / "results").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "1mi5_direct_new_further_0823.csv")
    print("Per-region coverage metrics (MAE/RMSE in kcal/mol):")
    print(df)


if __name__ == "__main__":
    main()
