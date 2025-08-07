import os
import glob
import MDAnalysis as mda

def split_structure(pdb_file):
    """
    Splits a PDB structure into two chains based on a linker sequence.
    Returns AtomGroup objects for each chain's atoms.
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
    
    # Return the full atom groups for the residues
    return chain_A_residues.atoms, chain_B_residues.atoms

def process_pdb(file_path, output_dir):
    """
    Processes a single PDB: splits it and writes the two chains to a specified output directory.
    """
    try:
        base_name = os.path.basename(file_path)
        print(f"Processing {base_name}...")
        chain_A, chain_B = split_structure(file_path)
        
        # Keep original filename and append _A/_B
        output_A_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_A.pdb")
        output_B_path = os.path.join(output_dir, f"{os.path.splitext(base_name)[0]}_B.pdb")
        
        chain_A.write(output_A_path)
        print(f"  -> Saved Chain A to {os.path.basename(output_A_path)}")
        
        chain_B.write(output_B_path)
        print(f"  -> Saved Chain B to {os.path.basename(output_B_path)}")

    except Exception as e:
        print(f"  -> Error processing {base_name}: {e}")

if __name__ == "__main__":
    # --- Configuration ---
    protein_root_dir = os.path.expanduser("~/Graphormer/distributional_graphormer/protein/")
    input_folder_name = "output_DMF5_adaptor_test5_centered_0805"
    pdb_directory = os.path.join(protein_root_dir, input_folder_name)
    
    # Define the output directory for split files, located inside the protein_root_dir
    output_dir_name = f"{input_folder_name}_split"
    output_directory = os.path.join(protein_root_dir, output_dir_name)
    # ---
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_directory, exist_ok=True)
    print(f"Output will be saved in: {output_directory}")

    if not os.path.isdir(pdb_directory):
        print(f"Input directory not found: {pdb_directory}")
    else:
        # Find all .pdb files
        pdb_files_to_process = glob.glob(os.path.join(pdb_directory, "*.pdb"))

        if pdb_files_to_process:
            print(f"\nFound {len(pdb_files_to_process)} PDB file(s) to split.")
            for pdb_file in pdb_files_to_process:
                # Check if split files already exist in the output directory
                base_name = os.path.basename(pdb_file)
                output_A_path = os.path.join(output_directory, f"{os.path.splitext(base_name)[0]}_A.pdb")
                if os.path.exists(output_A_path):
                    print(f"Skipping {base_name}, split files already exist.")
                    continue
                process_pdb(pdb_file, output_directory)
        else:
            print(f"No PDB files to split in {pdb_directory}")
            
    # Also process the reference PDB in the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    reference_pdb = os.path.join(current_dir, "DMF5_MD_init_linked_imgt.pdb")
    if os.path.exists(reference_pdb):
        ref_base_name = os.path.basename(reference_pdb)
        ref_output_A_path = os.path.join(output_directory, f"{os.path.splitext(ref_base_name)[0]}_A.pdb")

        if not os.path.exists(ref_output_A_path):
            print("\nProcessing reference PDB file...")
            process_pdb(reference_pdb, output_directory)
        else:
            print("\nReference PDB already split.")
