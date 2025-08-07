import numpy as np

fasta_file = "./A6_scTCR_linked_imgt.fasta"

# Create a per-residue mask of length 241 (residues 1-241)
linker_mask = np.zeros(241, dtype=np.uint8)

# Based on the FASTA file structure:
# Residues 1-110: TCR alpha chain
# Residues 111-125: Linker region (15 amino acids)
# Residues 126-241: TCR beta chain

# Set linker region (residues 111-125) to 1
linker_mask[110:125] = 1  # Python 0-indexed, so residues 111-125 are indices 110-124

# Save the mask
np.save('linker_mask.npy', linker_mask)
print('Saved linker_mask.npy with shape:', linker_mask.shape)
print('Number of linker residues:', np.sum(linker_mask))
print('Number of non-linker residues:', 241 - np.sum(linker_mask))
print('Linker region: residues', np.where(linker_mask == 1)[0] + 1)  # Convert back to 1-indexed