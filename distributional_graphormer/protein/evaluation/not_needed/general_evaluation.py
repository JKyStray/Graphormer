import numpy as np
import os
import glob
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from PDAnalysis.protein import Protein
from PDAnalysis.deformation import Deformation

def calculate_overall_rmsd_and_lddt():
    """
    Calculate per-structure overall CA RMSD and average lDDT for all residues,
    then save to general_evaluation.csv
    """

    # Paths
    current_dir   = os.path.dirname(os.path.abspath(__file__))
    reference_pdb = os.path.join(current_dir, "DMF5_MD_init_linked_imgt.pdb")
    output_dir    = os.path.join(os.path.dirname(current_dir), "output_dmf5_test3_0731")
    output_csv    = os.path.join(current_dir, "general_evaluation_dmf5.csv")
    
    # Sanity checks
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference structure not found: {reference_pdb}")
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")

    # Load reference CA atoms via MDAnalysis
    ref_u  = mda.Universe(reference_pdb)
    ref_ca = ref_u.select_atoms("name CA")
    ref_positions = ref_ca.positions

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
        mob_positions = mob_ca.positions
        rmsd_value = rms.rmsd(mob_positions, ref_positions,
                              center=True, superposition=True)
        print(f"  RMSD (overall CA): {rmsd_value:.4f} Å")

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
        # average over all residues
        avg_overall_lddt = float(np.mean(lddt_scores))
        print(f"  avg overall lDDT: {avg_overall_lddt:.4f}")

        results.append({
            'pdb_file':           name,
            'overall_ca_rmsd':    rmsd_value,
            'avg_overall_lddt':   avg_overall_lddt
        })

    # Write CSV
    if results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['pdb_file', 'overall_ca_rmsd', 'avg_overall_lddt']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        print(f"\nSaved results to {output_csv}")
    else:
        print("No valid PDBs processed.")

if __name__ == "__main__":
    calculate_overall_rmsd_and_lddt()
