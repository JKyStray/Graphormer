"""
In this file, we want to use TICA plot to evaluate the performance of the model. 

current directory: /home/jiaqi/Graphormer/distributional_graphormer/protein/evaluation/TICA

Data we have:
"ground truth": /home/jiaqi/Graphormer/distributional_graphormer/protein/dataset/Toyset_2/init_states
this contain about 10000 frames of MD trajectory of the A6 TCR. 
The frames are stored in npz file, which should be containing translation vector and rotation matrix. 

"prediction": /home/jiaqi/Graphormer/distributional_graphormer/protein/output_test3_0726
this contain 100 inference frames of the A6 TCR. For each frame, it has one final state npz file, 
one init state npz file and one pdb file. The model is a diffusion model, so the init states probably 
isn't useful for TICA plot. The final states and pdb files are the states that the model predicts. 
Do keep in mind that alignment might be needed. 

We want to plot the TICA plot for the ground truth and the prediction. 
Basically, we want to use ground truth to produce a TICA plot in contour, and then use scatter 
to plot the prediction, and therefore visualize the coverage of the prediction. 

Please keep this comment block in the code. 
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
from scipy.stats import gaussian_kde
import warnings
import pyemma
warnings.filterwarnings('ignore')

print("Using pyemma for TICA analysis")

def load_ground_truth_data(data_dir):
    """Load all ground truth trajectory data."""
    print("Loading ground truth data...")
    gt_files = glob.glob(os.path.join(data_dir, "init_state_*.npz"))
    
    all_translations = []
    all_rotations = []
    
    for i, file_path in enumerate(sorted(gt_files)):
        if i % 1000 == 0:
            print(f"Loading frame {i+1}/{len(gt_files)}")
        
        data = np.load(file_path)
        tr = data['tr']  # Shape: (241, 3)
        rot_mat = data['rot_mat']  # Shape: (241, 3, 3)
        
        # Scale MD data translation by 10 to match prediction scale
        tr = tr * 10.0
        
        all_translations.append(tr)
        all_rotations.append(rot_mat)
    
    print(f"Loaded {len(gt_files)} ground truth frames (translation scaled by 10)")
    return np.array(all_translations), np.array(all_rotations)

def load_prediction_data(data_dir):
    """Load prediction data from final states."""
    print("Loading prediction data...")
    pred_files = glob.glob(os.path.join(data_dir, "A6_*_final_state.npz"))
    
    all_translations = []
    all_rotations = []
    
    for file_path in sorted(pred_files):
        data = np.load(file_path)
        tr = data['tr']  # Shape: (241, 3)
        rot_mat = data['rot_mat']  # Shape: (241, 3, 3)
        
        all_translations.append(tr)
        all_rotations.append(rot_mat)
    
    print(f"Loaded {len(pred_files)} prediction frames")
    return np.array(all_translations), np.array(all_rotations)

def extract_features(translations, rotations):
    """Extract features from translation vectors and rotation matrices."""
    n_frames, n_residues, _ = translations.shape
    
    # Feature 1: Flatten translation vectors
    flat_translations = translations.reshape(n_frames, -1)
    
    # Feature 2: Extract rotation angles (convert rotation matrices to Euler angles)
    # For simplicity, use the trace of rotation matrices as a feature
    rotation_traces = np.trace(rotations, axis1=2, axis2=3)  # Shape: (n_frames, n_residues)
    
    # Feature 3: Center of mass
    center_of_mass = np.mean(translations, axis=1)  # Shape: (n_frames, 3)
    
    # Feature 4: Radius of gyration
    com_expanded = center_of_mass[:, np.newaxis, :]  # Shape: (n_frames, 1, 3)
    deviations = translations - com_expanded  # Shape: (n_frames, n_residues, 3)
    rg = np.sqrt(np.mean(np.sum(deviations**2, axis=2), axis=1))  # Shape: (n_frames,)
    
    # Combine features
    features = np.concatenate([
        flat_translations,
        rotation_traces,
        center_of_mass,
        rg.reshape(-1, 1)
    ], axis=1)
    
    return features

def align_structures(ref_coords, coords):
    """Simple Procrustes alignment to align coordinates to reference."""
    # Center both structures
    ref_centered = ref_coords - np.mean(ref_coords, axis=0)
    coords_centered = coords - np.mean(coords, axis=0)
    
    # Compute rotation matrix using SVD
    H = coords_centered.T @ ref_coords
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    
    # Ensure proper rotation (det(R) = 1)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T
    
    # Apply alignment
    aligned_coords = (R @ coords_centered.T).T + np.mean(ref_coords, axis=0)
    return aligned_coords

def apply_tica_analysis(gt_features, pred_features, lag_time=10):
    """Apply TICA analysis using pyemma."""
    print(f"Applying pyemma TICA with lag_time={lag_time}")
    
    # Use pyemma for TICA
    tica = pyemma.coordinates.tica(gt_features, lag=lag_time, dim=2)
    
    # Transform both datasets
    gt_tica = tica.get_output()[0]
    pred_tica = tica.transform(pred_features)
    
    return gt_tica, pred_tica

def create_tica_plot(gt_tica, pred_tica, save_path="tica_evaluation.png"):
    """Create TICA plot with contour for ground truth and scatter for predictions."""
    print("Creating TICA visualization...")
    
    # Close any existing plots to prevent conflicts
    plt.close('all')
    
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # Create density contour plot for ground truth
    x_gt, y_gt = gt_tica[:, 0], gt_tica[:, 1]
    
    # Calculate density using KDE
    try:
        kde = gaussian_kde(np.vstack([x_gt, y_gt]))
        
        # Create grid for contour plot
        x_min, x_max = x_gt.min(), x_gt.max()
        y_min, y_max = y_gt.min(), y_gt.max()
        
        # Extend range slightly
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_min -= 0.1 * x_range
        x_max += 0.1 * x_range
        y_min -= 0.1 * y_range
        y_max += 0.1 * y_range
        
        xx, yy = np.mgrid[x_min:x_max:100j, y_min:y_max:100j]
        positions = np.vstack([xx.ravel(), yy.ravel()])
        density = kde(positions).reshape(xx.shape)
        
        # Plot contours for ground truth
        contour = ax.contour(xx, yy, density, levels=8, colors='blue', alpha=0.6, linewidths=1.5)
        ax.contourf(xx, yy, density, levels=20, cmap='Blues', alpha=0.3)
        
    except Exception as e:
        print(f"KDE failed: {e}, using scatter plot for ground truth")
        ax.scatter(x_gt, y_gt, c='blue', alpha=0.1, s=1, label='Ground Truth')
    
    # Scatter plot for predictions
    x_pred, y_pred = pred_tica[:, 0], pred_tica[:, 1]
    scatter = ax.scatter(x_pred, y_pred, c='red', s=50, alpha=0.8, 
                        edgecolors='darkred', linewidths=1, label='Predictions')
    
    ax.set_xlabel('TICA Component 1', fontsize=12)
    ax.set_ylabel('TICA Component 2', fontsize=12)
    ax.set_title('TICA Analysis: Model Prediction Coverage', fontsize=14, fontweight='bold')
    
    # Create custom legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color='blue', lw=2, label='Ground Truth (Contour)'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
               markersize=8, label='Predictions (Scatter)', markeredgecolor='darkred')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Ensure the file is properly overwritten
    if os.path.exists(save_path):
        os.remove(save_path)
        print(f"Removed existing file: {save_path}")
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"TICA plot saved to: {save_path}")
    
    # Close the figure to free memory
    plt.close(fig)
    
    # Print coverage statistics
    print("\n=== Coverage Analysis ===")
    print(f"Ground truth TICA range: X[{x_gt.min():.3f}, {x_gt.max():.3f}], Y[{y_gt.min():.3f}, {y_gt.max():.3f}]")
    print(f"Prediction TICA range: X[{x_pred.min():.3f}, {x_pred.max():.3f}], Y[{y_pred.min():.3f}, {y_pred.max():.3f}]")
    
    # Calculate what fraction of predictions fall within ground truth convex hull
    from scipy.spatial import ConvexHull
    try:
        hull = ConvexHull(gt_tica)
        from matplotlib.path import Path
        hull_path = Path(gt_tica[hull.vertices])
        inside_hull = hull_path.contains_points(pred_tica)
        coverage_fraction = np.sum(inside_hull) / len(pred_tica)
        print(f"Predictions within ground truth convex hull: {np.sum(inside_hull)}/{len(pred_tica)} ({coverage_fraction:.1%})")
    except Exception as e:
        print(f"Could not calculate convex hull coverage: {e}")
    
    return fig

def main():
    """Main function to run TICA analysis."""
    print("Starting TICA Analysis for Model Evaluation")
    print("=" * 50)
    
    # Data paths
    gt_data_dir = "/home/jiaqi/Graphormer/distributional_graphormer/protein/dataset/Toyset_2/init_states"
    pred_data_dir = "/home/jiaqi/Graphormer/distributional_graphormer/protein/output_test3_0726"
    
    # Load data
    gt_translations, gt_rotations = load_ground_truth_data(gt_data_dir)
    pred_translations, pred_rotations = load_prediction_data(pred_data_dir)
    
    print(f"\nData loaded:")
    print(f"Ground truth: {gt_translations.shape[0]} frames")
    print(f"Predictions: {pred_translations.shape[0]} frames")
    
    # Optional: Align structures to first ground truth frame
    print("\nAligning structures...")
    ref_structure = gt_translations[0]  # Use first GT frame as reference
    
    # Align all ground truth frames
    aligned_gt_translations = np.zeros_like(gt_translations)
    for i, coords in enumerate(gt_translations):
        aligned_gt_translations[i] = align_structures(ref_structure, coords)
    
    # Align all prediction frames
    aligned_pred_translations = np.zeros_like(pred_translations)
    for i, coords in enumerate(pred_translations):
        aligned_pred_translations[i] = align_structures(ref_structure, coords)
    
    # Extract features
    print("\nExtracting features...")
    gt_features = extract_features(aligned_gt_translations, gt_rotations)
    pred_features = extract_features(aligned_pred_translations, pred_rotations)
    
    print(f"Feature dimensions: {gt_features.shape[1]}")
    
    # Apply TICA analysis
    print("\nApplying TICA analysis...")
    gt_tica, pred_tica = apply_tica_analysis(gt_features, pred_features, lag_time=10)
    
    # Create visualization with timestamp
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_filename = f"tica_evaluation_{timestamp}.png"
    fig = create_tica_plot(gt_tica, pred_tica, save_filename)
    
    print("\n" + "=" * 50)
    print("TICA Analysis Complete!")
    
    return gt_tica, pred_tica

if __name__ == "__main__":
    gt_tica, pred_tica = main()