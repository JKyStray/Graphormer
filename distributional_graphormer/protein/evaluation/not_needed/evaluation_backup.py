import numpy as np
import os
import glob
import csv
import MDAnalysis as mda
from MDAnalysis.analysis import rms
from PDAnalysis.protein import Protein
from PDAnalysis.deformation import Deformation

# Load masks
residue_mask = np.load("./residue_mask.npy")
linker_mask = np.load("./linker_mask.npy")

# residual_mask: 1 for CDR loop regions, 0 for Framework and linker region
# linker_mask: 1 for linker region, 0 otherwise

# framework_mask: 1 for framework region, 0 for CDR loop and linker region
framework_mask = np.logical_and(residue_mask == 0, linker_mask == 0).astype(np.uint8)

def calculate_framework_rmsd():
    """
    Calculate C alpha atom RMSD for framework regions between inference outcomes
    and reference structure, then save to evaluation.csv
    """

    # Paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_pdb = os.path.join(current_dir, "MD_init_linked_imgt.pdb")
    output_dir = os.path.join(os.path.dirname(current_dir), "output_test1_0726")
    output_csv = os.path.join(current_dir, "evaluation_test1.csv")
    
    # Check if reference file exists
    if not os.path.exists(reference_pdb):
        raise FileNotFoundError(f"Reference structure not found: {reference_pdb}")
    
    if not os.path.exists(output_dir):
        raise FileNotFoundError(f"Output directory not found: {output_dir}")
    
    # Load reference structure
    print(f"Loading reference structure: {reference_pdb}")
    ref_universe = mda.Universe(reference_pdb)
    ref_ca = ref_universe.select_atoms("name CA")
    
    # Check mask length matches CA atoms
    if len(ref_ca) != len(framework_mask):
        raise ValueError(f"Framework mask length ({len(framework_mask)}) doesn't match "
                        f"number of CA atoms ({len(ref_ca)}) in reference structure")
    
    # Get framework CA positions from reference
    fw_bool = framework_mask.astype(bool)
    ref_fw_positions = ref_ca.positions[fw_bool]
    
    print(f"Framework mask: {np.sum(framework_mask)} residues out of {len(framework_mask)} total")
    print(f"Reference CA atoms: {len(ref_ca)}")
    print(f"Framework CA atoms: {len(ref_fw_positions)}")
    
    # Find all PDB files in output directory
    pdb_files = glob.glob(os.path.join(output_dir, "*.pdb"))
    # Sort pdb_files by the integer index in the filename (e.g., "A6_123.pdb" -> 123)
    def extract_index(pdb_path):
        base = os.path.basename(pdb_path)
        # Assumes format "A6_<number>.pdb"
        try:
            idx_str = base.split('_')[1].split('.')[0]
            return int(idx_str)
        except Exception:
            return float('inf')  # Put any non-matching files at the end

    pdb_files = sorted(pdb_files, key=extract_index)
    
    print(f"Found {len(pdb_files)} PDB files to process")
    
    results = []
    
    for pdb_file in pdb_files:
        filename = os.path.basename(pdb_file)
        print(f"Processing: {filename}")
        
        try:
            # Load inference outcome structure
            mob_universe = mda.Universe(pdb_file)
            mob_ca = mob_universe.select_atoms("name CA")
            
            # Check if number of CA atoms matches reference
            if len(mob_ca) != len(ref_ca):
                print(f"Warning: {filename} has {len(mob_ca)} CA atoms, expected {len(ref_ca)}")
                continue
            
            # Get framework CA positions from mobile structure
            mob_fw_positions = mob_ca.positions[fw_bool]
            
            # Calculate RMSD with optimal superposition
            rmsd_value = rms.rmsd(mob_fw_positions, ref_fw_positions, center=True, superposition=True)
            
            results.append({
                'pdb_file': filename,
                'framework_ca_rmsd': rmsd_value
            })
            
            print(f"  RMSD: {rmsd_value:.4f} Å")
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")
            continue
    
    # Save results to CSV
    if results:
        with open(output_csv, 'w', newline='') as csvfile:
            fieldnames = ['pdb_file', 'framework_ca_rmsd']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"\nResults saved to: {output_csv}")
        print(f"Processed {len(results)} structures successfully")
        
        # Print summary statistics
        rmsd_values = np.array([r['framework_ca_rmsd'] for r in results])
        print(f"\nSummary statistics:")
        print(f"  Mean RMSD: {np.mean(rmsd_values):.4f} Å")
        print(f"  Std RMSD:  {np.std(rmsd_values):.4f} Å")
        print(f"  Min RMSD:  {np.min(rmsd_values):.4f} Å")
        print(f"  Max RMSD:  {np.max(rmsd_values):.4f} Å")
    else:
        print("No results to save.")

if __name__ == "__main__":
    calculate_framework_rmsd()

