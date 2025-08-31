import torch
import click
import numpy as np

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
    N = torch.matmul(rot_mat.transpose(-1, -2), N_ref) + CA
    C = torch.matmul(rot_mat.transpose(-1, -2), C_ref) + CA
    return CA, N, C


def write_to_pdb(seq, tr, rot_mat, file):
    CA, N, C = convert_to_CANC(tr, rot_mat)
    with open(file, "w") as fp:
        lines = xyz2pdb(seq, CA, N, C)
        fp.write("\n".join(lines))

@click.command()
@click.option("-f", "--fasta", default="./dataset/1mi5_pseudo_single.fasta", required=True, help="Input FASTA file")
@click.option("-n", "--npz", required=True, help="Input NPZ file with tr and rot_mat")
@click.option("-o", "--output", required=True, help="Output PDB file")
def main(fasta, npz, output):
    """
    Generates a PDB file from a FASTA sequence and a NPZ file containing translation and rotation data.
    """
    with open(fasta, "r") as f:
        lines = f.readlines()
        seq = "".join([l.strip() for l in lines[1:]])

    data = np.load(npz)
    tr = torch.from_numpy(data["tr"]).float()
    rot_mat = torch.from_numpy(data["rot_mat"]).float()

    write_to_pdb(seq, tr, rot_mat, output)
    print(f"Successfully wrote PDB file to {output}")

if __name__ == "__main__":
    main()
