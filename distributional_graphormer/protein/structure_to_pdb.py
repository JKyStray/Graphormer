import torch
import click
import numpy as np
import os
from glob import glob

def xyz2pdb(seq, CA, N, C):
    one_to_three = {
        "A": "ALA",
        "C": "CYS",
        "D": "ASP",
        "E": "GLU",
        "F": "PHE",
        "G": "GLY",
        "H": "HIS",
        "I": "ILE",
        "K": "LYS",
        "L": "LEU",
        "M": "MET",
        "N": "ASN",
        "P": "PRO",
        "Q": "GLN",
        "R": "ARG",
        "S": "SER",
        "T": "THR",
        "V": "VAL",
        "W": "TRP",
        "Y": "TYR",
        "X": "UNK",
    }
    line = "ATOM%7i  %s  %s A%4i    %8.3f%8.3f%8.3f  1.00  0.00           C"
    ret = []
    for i in range(CA.shape[0]):
        ret.append(
            line
            % (
                3 * i + 1,
                "CA",
                one_to_three[seq[i]],
                i + 1,
                CA[i][0],
                CA[i][1],
                CA[i][2],
            )
        )
        ret.append(
            line
            % (3 * i + 2, " C", one_to_three[seq[i]], i + 1, C[i][0], C[i][1], C[i][2])
        )
        ret.append(
            line
            % (3 * i + 3, " N", one_to_three[seq[i]], i + 1, N[i][0], N[i][1], N[i][2])
        )
    ret.append("TER")
    return ret

def convert_to_CANC(tr, rot_mat):
    tr, rot_mat = tr.cpu(), rot_mat.cpu()
    CA = tr * 10
    N_ref = torch.tensor([1.45597958, 0.0, 0.0])
    C_ref = torch.tensor([-0.533655602, 1.42752619, 0.0])
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref.unsqueeze(-1)).squeeze(-1) + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref.unsqueeze(-1)).squeeze(-1) + CA
    return CA, N, C


def write_to_pdb(seq, tr, rot_mat, file):
    CA, N, C = convert_to_CANC(tr, rot_mat)
    with open(file, "w") as fp:
        lines = xyz2pdb(seq, CA, N, C)
        fp.write("\n".join(lines))

@click.command()
@click.option("-i", "--input_dir", default="./dataset/Mars2/raw/1mi5/structure", required=True, help="Input directory with npz structure files")
@click.option("-o", "--output_dir", default="./dataset/Mars2/raw/1mi5/pdb_structure", required=True, help="Output directory for PDB files")
@click.option("-f", "--fasta", default="dataset/1mi5_pseudo_single.fasta", required=True, help="Input FASTA file for sequence")
def main(input_dir, output_dir, fasta):
    """
    Converts all NPZ structure files in a directory to PDB files.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        with open(fasta, "r") as f:
            lines = f.readlines()
            seq = "".join([l.strip() for l in lines[1:]])
    except FileNotFoundError:
        print(f"Error: FASTA file not found at {fasta}")
        return
    
    npz_files = glob(os.path.join(input_dir, '*.npz'))
    
    if not npz_files:
        print(f"No .npz files found in {input_dir}")
        return

    for npz_file in npz_files:
        try:
            data = np.load(npz_file)
            tr = torch.from_numpy(data["tr"]).float()
            rot_mat = torch.from_numpy(data["rot_mat"]).float()
            
            base_name = os.path.basename(npz_file)
            output_filename = os.path.splitext(base_name)[0] + ".pdb"
            output_path = os.path.join(output_dir, output_filename)

            write_to_pdb(seq, tr, rot_mat, output_path)
            print(f"Successfully wrote PDB file to {output_path}")

        except KeyError:
            print(f"Skipping {npz_file}: 'tr' or 'rot_mat' not found.")
        except Exception as e:
            print(f"Failed to process {npz_file}: {e}")

if __name__ == "__main__":
    main()
