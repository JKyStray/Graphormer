import numpy as np
import os
import glob
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from PDAnalysis.protein import Protein
from PDAnalysis.deformation import Deformation

# Load masks
# I precomputed the masks with length 241 residues instead of all atoms

residue_mask = np.load("./residue_mask.npy") # 1 for CDR loop regions, 0 for Framework and linker region
linker_mask   = np.load("./linker_mask.npy") # 1 for linker region, 0 otherwise

# framework_mask: 1 for framework region, 0 otherwise
framework_mask = np.logical_and(residue_mask == 0, linker_mask == 0).astype(np.uint8)
fw_bool = framework_mask.astype(bool)

def calculate_framework_rmsd_and_lddt():
    """
    Calculate per-structure framework-only CA RMSD and average lDDT,
    then save to evaluation.csv
    """

    # Paths
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    reference_pdb = os.path.join(current_dir, "MD_init_linked_imgt.pdb")
    output_dir    = os.path.join(os.path.dirname(current_dir), "output_evo2_embedding")
    output_csv    = os.path.join(current_dir, "evaluation3.csv")
    
    # Sanity checks
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference structure not found: {reference_pdb}")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Load reference CA atoms via MDAnalysis
    ref_u  = mda.Universe(reference_pdb)
    ref_ca = ref_u.select_atoms("name CA")
    if len(ref_ca) != len(framework_mask):
        raise ValueError(f"Mask length ({len(framework_mask)}) != CA count ({len(ref_ca)})")
    ref_fw_positions = ref_ca.positions[fw_bool]

    # Prepare list of output PDBs
    pdb_files = sorted(
        glob.glob(os.path.join(output_dir, "*.pdb")),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]) 
                     if '_' in os.path.basename(p) else float('inf')
    )

    results = []
    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file)
        print(f"Processing {name}...")

        # --- RMSD ---
        mob_u  = mda.Universe(pdb_file)
        mob_ca = mob_u.select_atoms("name CA")
        if len(mob_ca) != len(ref_ca):
            print(f"  ⚠️ {name} has {len(mob_ca)} CA atoms, expected {len(ref_ca)}")
            continue
        mob_fw_positions = mob_ca.positions[fw_bool]
        rmsd_value = rms.rmsd(mob_fw_positions, ref_fw_positions,
                              center=True, superposition=True)
        print(f"  RMSD (framework CA): {rmsd_value:.4f} Å")

        # --- lDDT ---
        # include all residues (min_plddt=0, max_bfactor=inf), neigh_cut at 13 Å
        prot_ref = Protein(reference_pdb,
                           min_plddt=0.0,
                           max_bfactor=float('inf'),
                           neigh_cut=13.0)
        prot_mob = Protein(pdb_file,
                           min_plddt=0.0,
                           max_bfactor=float('inf'),
                           neigh_cut=13.0)
        deform = Deformation(prot_ref, prot_mob, method="lddt")
        deform.run()
        lddt_scores = deform.lddt
        # average over framework residues
        avg_fw_lddt = float(np.mean(lddt_scores[fw_bool]))
        print(f"  avg framework lDDT: {avg_fw_lddt:.4f}")

        results.append({
            'pdb_file':           name,
            'framework_ca_rmsd':  rmsd_value,
            'avg_framework_lddt': avg_fw_lddt
        })

    # Write CSV
    if results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['pdb_file', 'framework_ca_rmsd', 'avg_framework_lddt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {output_csv}")
    else:
        print("No valid PDBs processed.")

if __name__ == "__main__":
    calculate_framework_rmsd_and_lddt()
