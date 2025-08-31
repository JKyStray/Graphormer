import numpy as np
from pathlib import Path

def convert_binary_to_labeled_cdr_mask(binary_mask_path: str, output_path: str = None) -> np.ndarray:
    """
    Convert a binary CDR mask to separate labels for each CDR region.
    
    For TCR: alpha chain + linker + beta chain
    Expected order: CDR1α, CDR2α, CDR3α, [linker], CDR1β, CDR2β, CDR3β
    Labels: 1, 2, 3, 4, 5, 6 respectively
    
    Args:
        binary_mask_path: Path to binary mask (0=framework, 1=CDR)
        output_path: Optional path to save labeled mask
    
    Returns:
        labeled_mask: Array with labels 1-6 for each CDR region
    """
    # Load binary mask
    binary_mask = np.load(binary_mask_path)
    print(f"Binary mask shape: {binary_mask.shape}")
    print(f"Total CDR residues: {np.sum(binary_mask)}")
    
    # Find CDR regions (contiguous segments of 1s)
    labeled_mask = np.zeros_like(binary_mask, dtype=int)
    
    # Find transitions from 0 to 1 (CDR start) and 1 to 0 (CDR end)
    diff = np.diff(np.concatenate([[0], binary_mask, [0]]))
    starts = np.where(diff == 1)[0]  # CDR region starts
    ends = np.where(diff == -1)[0]   # CDR region ends
    
    print(f"\nFound {len(starts)} CDR regions:")
    
    cdr_labels = ["CDR1α", "CDR2α", "CDR3α", "CDR1β", "CDR2β", "CDR3β"]
    
    if len(starts) != 6:
        print(f"WARNING: Expected 6 CDR regions, found {len(starts)}")
        print("This might indicate:")
        print("- Some CDRs are missing from the structure")
        print("- Some CDRs are merged (too close together)")
        print("- The linker region might not be clearly separated")
        print("\nProceeding with available regions...")
    
    # Label each region
    for i, (start, end) in enumerate(zip(starts, ends)):
        length = end - start
        label = i + 1  # Labels 1-6
        labeled_mask[start:end] = label
        
        region_name = cdr_labels[i] if i < len(cdr_labels) else f"CDR{label}"
        print(f"  {region_name}: positions {start}-{end-1} (length {length})")
    
    # Verify the conversion
    print(f"\nVerification:")
    print(f"Original CDR residues: {np.sum(binary_mask)}")
    print(f"Labeled CDR residues: {np.sum(labeled_mask > 0)}")
    print(f"Unique labels: {np.unique(labeled_mask)}")
    
    # Check for each label
    for i in range(1, 7):
        count = np.sum(labeled_mask == i)
        region_name = cdr_labels[i-1] if i <= len(cdr_labels) else f"CDR{i}"
        print(f"  {region_name}: {count} residues")
    
    # Save if output path provided
    if output_path:
        output_path = Path(output_path).expanduser().resolve()
        np.save(output_path, labeled_mask)
        print(f"\nSaved labeled mask to: {output_path}")
    
    return labeled_mask

def analyze_cdr_distribution(labeled_mask: np.ndarray) -> None:
    """
    Analyze the distribution and spacing of CDR regions.
    """
    print("\n" + "="*50)
    print("CDR DISTRIBUTION ANALYSIS")
    print("="*50)
    
    # Find positions of each CDR
    for label in range(1, 7):
        positions = np.where(labeled_mask == label)[0]
        if len(positions) > 0:
            start, end = positions[0], positions[-1]
            length = len(positions)
            
            # Determine chain
            chain = "Alpha" if label <= 3 else "Beta"
            cdr_num = label if label <= 3 else label - 3
            
            print(f"CDR{cdr_num} ({chain}): {start:3d}-{end:3d} ({length:2d} residues)")
    
    # Check for gaps between alpha and beta chains
    alpha_end = np.where(labeled_mask == 3)[0]
    beta_start = np.where(labeled_mask == 4)[0]
    
    if len(alpha_end) > 0 and len(beta_start) > 0:
        linker_length = beta_start[0] - alpha_end[-1] - 1
        print(f"\nLinker region: {alpha_end[-1]+1}-{beta_start[0]-1} ({linker_length} residues)")

# Example usage
if __name__ == "__main__":
    # Edit this path to your actual mask file
    binary_mask_path = "~/Graphormer/distributional_graphormer/protein/evaluation/a6_cdr_mask.npy"
    output_path = "~/Graphormer/distributional_graphormer/protein/evaluation/a6_cdr_mask_labeled.npy"
    
    # Convert binary mask to labeled mask
    labeled_mask = convert_binary_to_labeled_cdr_mask(binary_mask_path, output_path)
    
    # Analyze the distribution
    analyze_cdr_distribution(labeled_mask)
    
    print("\n" + "="*50)
    print("USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Review the CDR positions above to ensure they make sense")
    print("2. If positions look wrong, you may need to manually adjust")
    print("3. Use the labeled mask in your coverage evaluation:")
    print(f"   cdr_mask_path = '{output_path}'")
    print("4. The coverage script will now evaluate each CDR separately")