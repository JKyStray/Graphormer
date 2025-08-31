import click
import numpy as np


def calc_rotate_imat(N, CA, C):
    """
    Calculate rotation matrix from N, CA, C coordinates
    """
    p1 = N - CA
    x = p1 / np.linalg.norm(p1, axis=-1, keepdims=True)
    p2 = C - N
    inner_1 = np.matmul(np.expand_dims(p1, axis=1), np.expand_dims(p1, axis=2))[:, :, 0]
    inner_2 = np.matmul(np.expand_dims(-p1, axis=1), np.expand_dims(p2, axis=2))[
        :, :, 0
    ]
    alpha = inner_1 / inner_2
    y = alpha * p2 + p1
    y = y / np.linalg.norm(y, axis=-1, keepdims=True)
    z = np.cross(x, y)
    mat = np.concatenate([x, y, z], axis=-1)
    mat = mat.reshape(*mat.shape[:-1], 3, 3)
    return mat


def parse_pdb(pdb_file):
    """
    A simple PDB parser to extract N, CA, C coordinates
    """
    with open(pdb_file, 'r') as f:
        lines = f.readlines()

    coords = {'N': [], 'CA': [], 'C': []}
    
    for line in lines:
        if line.startswith('ATOM'):
            atom_name = line[12:16].strip()
            if atom_name in ['N', 'CA', 'C']:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                coords[atom_name].append([x, y, z])

    N = np.array(coords['N'])
    CA = np.array(coords['CA'])
    C = np.array(coords['C'])
    
    return N, CA, C

@click.command()
@click.option('--pdb_file', help='input pdb file')
@click.option('--out_file', help='output npz file')
def main(pdb_file, out_file):
    N, CA, C = parse_pdb(pdb_file)
    
    center = np.mean(CA, axis=0)
    
    N -= center
    CA -= center
    C -= center
    
    tr = CA
    rot_mat = calc_rotate_imat(N, CA, C)
    
    np.savez(out_file, tr=tr, rot_mat=rot_mat)

if __name__ == '__main__':
    main()

