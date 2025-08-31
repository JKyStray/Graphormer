"""
The script below is used to calculate the rmsd for each of the two chains separately. 

We want to make a change here: we want to calculate the rmsd for the framework region of 
each of the two chains. 

The framework region is the residues that are not CDR loop nor linker region. 

The framework mask, as example, is 
~/Graphormer/distributional_graphormer/protein/evaluation/a6_framework_mask.npy

An idea is to split the mask as well...

For you to understand the structure, the reference pdb is 
~/Graphormer/distributional_graphormer/protein/evaluation/A6_MD_init_linked_imgt.pdb, 

The inference output to evaluate is in
~/Graphormer/distributional_graphormer/protein/output_a6_adaptor_test5_centered_0805/

For the output, we want to have the following columns: 
- pdb_file
- original_framework_rmsd
- A_chain_framework_rmsd
- B_chain_framework_rmsd
- total_framework_rmsd

where original_framework_rmsd is the rmsd of the framework region of the whole structure without split. 
total_framework_rmsd is A_chain_framework_rmsd + B_chain_framework_rmsd. 

Please keep this comment block. 
"""

import numpy as np
import os
import glob
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def split_structure(pdb_file, framework_mask):
    """
    Splits a PDB structure into two chains based on a linker sequence,
    and applies a framework mask.
    Returns the C-alpha atoms for the framework region of each chain.
    """
    u = mda.Universe(pdb_file)
    
    # Ensure the number of residues in PDB matches the mask length
    if len(u.residues) != len(framework_mask):
        raise ValueError(f"Mismatch between number of residues in {os.path.basename(pdb_file)} ({len(u.residues)}) and framework mask length ({len(framework_mask)})")
        
    resnames = list(u.residues.resnames)
    
    linker_resnames = ['GLY', 'GLY', 'GLY', 'GLY', 'SER', 
                       'GLY', 'GLY', 'GLY', 'GLY', 'SER', 
                       'GLY', 'GLY', 'GLY', 'GLY', 'SER']
    
    linker_start_index = -1
    for i in range(len(resnames) - len(linker_resnames) + 1):
        if resnames[i:i+len(linker_resnames)] == linker_resnames:
            linker_start_index = i
            break
            
    if linker_start_index == -1:
        raise ValueError(f"Linker not found in {os.path.basename(pdb_file)}")
        
    # Split the framework mask
    mask_A = framework_mask[:linker_start_index]
    mask_B = framework_mask[linker_start_index + len(linker_resnames):]

    # Get residues for each chain and apply the mask
    chain_A_residues = u.residues[:linker_start_index][mask_A]
    chain_B_residues = u.residues[linker_start_index + len(linker_resnames):][mask_B]
    
    chain_A_ca = chain_A_residues.atoms.select_atoms("name CA")
    chain_B_ca = chain_B_residues.atoms.select_atoms("name CA")
    
    return chain_A_ca, chain_B_ca

def calculate_rmsd_for_chains(reference_A_ca, reference_B_ca, pdb_files_dir, output_csv_path, reference_pdb_path, framework_mask):
    """
    Calculates RMSD for the framework region of each chain and the total structure.
    """
    
    ref_u = mda.Universe(reference_pdb_path)
    ref_framework_ca = ref_u.residues[framework_mask].atoms.select_atoms("name CA")

    results = []
    
    pdb_files = sorted(
        glob.glob(os.path.join(pdb_files_dir, "*.pdb")),
        key=lambda p: int(os.path.basename(p).split('_')[1].split('.')[0]) 
                     if '_' in os.path.basename(p) and len(os.path.basename(p).split('_')) > 1 and os.path.basename(p).split('_')[1].split('.')[0].isdigit() 
                     else float('inf')
    )

    if not pdb_files:
        print(f"No PDB files found in {pdb_files_dir}")
        return

    for pdb_file in pdb_files:
        name = os.path.basename(pdb_file)
        print(f"Processing {name}...")
        
        try:
            mob_u = mda.Universe(pdb_file)
            
            # Original Framework RMSD
            mob_framework_ca = mob_u.residues[framework_mask].atoms.select_atoms("name CA")
            if len(mob_framework_ca) != len(ref_framework_ca):
                print(f"  ⚠️  {name} has {len(mob_framework_ca)} framework CA atoms, expected {len(ref_framework_ca)}. Skipping original framework RMSD.")
                original_framework_rmsd = np.nan
            else:
                original_framework_rmsd = rms.rmsd(mob_framework_ca.positions, ref_framework_ca.positions, center=True, superposition=True)
                print(f"  RMSD (Original Framework): {original_framework_rmsd:.4f} Å")
            
            # Split mobile structure using the framework mask
            mob_A_ca, mob_B_ca = split_structure(pdb_file, framework_mask)

            # RMSD for chain A framework
            if len(mob_A_ca) != len(reference_A_ca):
                print(f"  ⚠️  {name} chain A framework has {len(mob_A_ca)} CA atoms, expected {len(reference_A_ca)}. Skipping chain A framework RMSD.")
                rmsd_A = np.nan
            else:
                rmsd_A = rms.rmsd(mob_A_ca.positions, reference_A_ca.positions, center=True, superposition=True)
                print(f"  RMSD (Chain A Framework): {rmsd_A:.4f} Å")

            # RMSD for chain B framework
            if len(mob_B_ca) != len(reference_B_ca):
                print(f"  ⚠️  {name} chain B framework has {len(mob_B_ca)} CA atoms, expected {len(reference_B_ca)}. Skipping chain B framework RMSD.")
                rmsd_B = np.nan
            else:
                rmsd_B = rms.rmsd(mob_B_ca.positions, reference_B_ca.positions, center=True, superposition=True)
                print(f"  RMSD (Chain B Framework): {rmsd_B:.4f} Å")

            total_framework_rmsd = np.nansum([rmsd_A, rmsd_B])

            results.append({
                'pdb_file': name,
                'original_framework_rmsd': original_framework_rmsd,
                'A_chain_framework_rmsd': rmsd_A,
                'B_chain_framework_rmsd': rmsd_B,
                'total_framework_rmsd': total_framework_rmsd
            })

        except Exception as e:
            print(f"  Could not process {name}. Error: {e}")
            
    # Write CSV
    if results:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['pdb_file', 'original_framework_rmsd', 'A_chain_framework_rmsd', 'B_chain_framework_rmsd', 'total_framework_rmsd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                for key, value in row.items():
                    if isinstance(value, float) and not np.isnan(value):
                        row[key] = f"{value:.4f}"
                writer.writerow(row)
        print(f"\nSaved results to {output_csv_path}")
    else:
        print("No valid PDBs processed.")

if __name__ == "__main__":
    reference_name = "1mi5"
    inference_folder_name = "output_1mi5_rm29_evo_0821"
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    # Use paths from the comment block
    reference_pdb = os.path.expanduser(f"~/Graphormer/distributional_graphormer/protein/evaluation/{reference_name}_MD_init_linked_imgt.pdb")
    framework_mask_path = os.path.expanduser(f"~/Graphormer/distributional_graphormer/protein/evaluation/{reference_name.lower()}_framework_mask.npy")
    
    pdb_files_dir = os.path.expanduser(f"~/Graphormer/distributional_graphormer/protein/{inference_folder_name}/")
    
    output_csv = os.path.join(current_dir, f"../csv_output/evaluation_{inference_folder_name}_framework_separate.csv")

    if not os.path.exists(pdb_files_dir):
        raise FileNotFoundError(f"Input directory not found: {pdb_files_dir}")
        
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")

    if not os.path.exists(framework_mask_path):
        raise FileNotFoundError(f"Framework mask not found: {framework_mask_path}")

    print("Loading framework mask...")
    framework_mask = np.load(framework_mask_path).astype(bool)
    print(f"Framework mask loaded. Number of framework residues: {np.sum(framework_mask)}")

    print("Splitting reference PDB...")
    try:
        reference_A_ca, reference_B_ca = split_structure(reference_pdb, framework_mask)
        print(f"Reference Chain A CA atoms: {len(reference_A_ca)}")
        print(f"Reference Chain B CA atoms: {len(reference_B_ca)}")
        
        calculate_rmsd_for_chains(reference_A_ca, reference_B_ca, pdb_files_dir, output_csv, reference_pdb, framework_mask)
        
    except ValueError as e:
        print(f"Error splitting reference PDB: {e}")
