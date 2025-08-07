import numpy as np

# Load the atom-level mask
atom_mask = np.load('a6_all_atom_mask.npy')

# Parse the PDB to get residue indices for each atom
pdb_file = 'MD_init_linked_imgt.pdb'
residue_indices = []
with open(pdb_file, 'r') as f:
    for line in f:
        if line.startswith('ATOM'):
            res_seq = int(line[22:26])  # Residue sequence number
            residue_indices.append(res_seq)

residue_indices = np.array(residue_indices)

# Map residue index to CDR status (1 if any atom in residue is CDR, else 0)
residue_to_cdr = {}
for idx, res_seq in enumerate(residue_indices):
    if res_seq not in residue_to_cdr:
        residue_to_cdr[res_seq] = atom_mask[idx]
    else:
        residue_to_cdr[res_seq] = max(residue_to_cdr[res_seq], atom_mask[idx])

# Create a per-residue mask of length 241 (residues 1-241)
residual_mask = np.zeros(241, dtype=np.uint8)
for res_seq in range(1, 242):  # Residues 1 to 241
    if res_seq in residue_to_cdr:
        residual_mask[res_seq - 1] = residue_to_cdr[res_seq]

# Save the new mask
np.save('residual_mask.npy', residual_mask)
print('Saved residual_mask.npy with shape:', residual_mask.shape)
print('Number of CDR residues:', np.sum(residual_mask))
print('Number of framework/linker residues:', 241 - np.sum(residual_mask))
