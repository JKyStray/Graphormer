"""
The script below is currently used to calculate overall rmsd and lDDT. 

However, we want to make a change here: 

Since the structure is actually a complex, linked with linker "GGGGSGGGGSGGGGS", we want to 
calculate the rmsd for each of the two chains separately. 

That is, for input pdb files and reference pdb, we want to first split the structure into two chains, 
then calculate the rmsd for each chain separately. 

We can delete the lDDT part for now. 

The output should be a csv file with the following columns: 
- pdb_file
- A_chain_overall_rmsd
- B_chain_overall_rmsd
- total_rmsd

For you to understand the structure, the reference pdb is 
~/Graphormer/distributional_graphormer/protein/evaluation/A6_MD_init_linked_imgt.pdb, 

The inference output to evaluate is in
~/Graphormer/distributional_graphormer/protein/output_test3_0726/

Please keep this comment block. 
"""

import numpy as np
import os
import glob
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import rms

def split_structure(pdb_file):
    """
    Splits a PDB structure into two chains based on a linker sequence.
    Returns the C-alpha atoms for each chain.
    """
    u = mda.Universe(pdb_file)
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
        
    chain_A_residues = u.residues[:linker_start_index]
    chain_B_residues = u.residues[linker_start_index + len(linker_resnames):]
    
    chain_A_ca = chain_A_residues.atoms.select_atoms("name CA")
    chain_B_ca = chain_B_residues.atoms.select_atoms("name CA")
    
    return chain_A_ca, chain_B_ca

def calculate_rmsd_for_chains(reference_A_ca, reference_B_ca, pdb_files_dir, output_csv_path, reference_pdb_path):
    """
    Calculates RMSD for each chain and the total structure.
    """
    
    ref_u = mda.Universe(reference_pdb_path)
    ref_ca = ref_u.select_atoms("name CA")

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
            
            # Total RMSD
            mob_ca = mob_u.select_atoms("name CA")
            if len(mob_ca) != len(ref_ca):
                print(f"  ⚠️  {name} has {len(mob_ca)} CA atoms, expected {len(ref_ca)} for total structure. Skipping.")
                continue

            total_rmsd = rms.rmsd(mob_ca.positions, ref_ca.positions, center=True, superposition=True)
            print(f"  RMSD (total CA): {total_rmsd:.4f} Å")
            
            # Split mobile structure
            mob_A_ca, mob_B_ca = split_structure(pdb_file)

            # RMSD for chain A
            if len(mob_A_ca) != len(reference_A_ca):
                print(f"  ⚠️  {name} chain A has {len(mob_A_ca)} CA atoms, expected {len(reference_A_ca)}. Skipping chain A RMSD.")
                rmsd_A = np.nan
            else:
                rmsd_A = rms.rmsd(mob_A_ca.positions, reference_A_ca.positions, center=True, superposition=True)
                print(f"  RMSD (Chain A CA): {rmsd_A:.4f} Å")

            # RMSD for chain B
            if len(mob_B_ca) != len(reference_B_ca):
                print(f"  ⚠️  {name} chain B has {len(mob_B_ca)} CA atoms, expected {len(reference_B_ca)}. Skipping chain B RMSD.")
                rmsd_B = np.nan
            else:
                rmsd_B = rms.rmsd(mob_B_ca.positions, reference_B_ca.positions, center=True, superposition=True)
                print(f"  RMSD (Chain B CA): {rmsd_B:.4f} Å")

            results.append({
                'pdb_file': name,
                'original_rmsd': total_rmsd,
                'A_chain_overall_rmsd': rmsd_A,
                'B_chain_overall_rmsd': rmsd_B,
                'total_rmsd': rmsd_A + rmsd_B
            })

        except Exception as e:
            print(f"  Could not process {name}. Error: {e}")
            
    # Write CSV
    if results:
        with open(output_csv_path, 'w', newline='') as csvfile:
            fieldnames = ['pdb_file', 'original_rmsd', 'A_chain_overall_rmsd', 'B_chain_overall_rmsd', 'total_rmsd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for row in results:
                # Format floats for CSV
                if not np.isnan(row['A_chain_overall_rmsd']):
                    row['A_chain_overall_rmsd'] = f"{row['A_chain_overall_rmsd']:.4f}"
                if not np.isnan(row['B_chain_overall_rmsd']):
                    row['B_chain_overall_rmsd'] = f"{row['B_chain_overall_rmsd']:.4f}"
                row['total_rmsd'] = f"{row['total_rmsd']:.4f}"
                writer.writerow(row)
        print(f"\nSaved results to {output_csv_path}")
    else:
        print("No valid PDBs processed.")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, '..'))

    # Use paths from the comment block
    reference_pdb = os.path.expanduser("~/Graphormer/distributional_graphormer/protein/evaluation/DMF5_MD_init_linked_imgt.pdb")
    
    inference_folder_name = "output_DMF5_adaptor_test5_centered_0805"
    pdb_files_dir = os.path.expanduser(f"~/Graphormer/distributional_graphormer/protein/{inference_folder_name}/")
    
    output_csv = os.path.join(current_dir, f"evaluation_{inference_folder_name}_new.csv")

    if not os.path.exists(pdb_files_dir):
        raise FileNotFoundError(f"Input directory not found: {pdb_files_dir}")
        
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference PDB not found: {reference_pdb}")

    print("Splitting reference PDB...")
    try:
        reference_A_ca, reference_B_ca = split_structure(reference_pdb)
        print(f"Reference Chain A CA atoms: {len(reference_A_ca)}")
        print(f"Reference Chain B CA atoms: {len(reference_B_ca)}")
        
        calculate_rmsd_for_chains(reference_A_ca, reference_B_ca, pdb_files_dir, output_csv, reference_pdb)
        
    except ValueError as e:
        print(f"Error splitting reference PDB: {e}")
