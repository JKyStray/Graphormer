import math

import numpy as np
import torch
import torch.nn.functional as F
from common import config as cfg
from torch import nn

from . import geometry, so3
from .base_model import BaseModel
from .positional_encoding import RelativePositionBias
from .structure_module import StructureModule


class SinusoidalPositionEmbeddings(nn.Module):
    """
    Implements sinusoidal position embeddings similar to those used in the Transformer architecture.
    These embeddings help the model understand the temporal dimension in the diffusion process.
    """
    def __init__(
        self,
        dim,
        max_period=10000,
    ):
        super().__init__()
        self.dim = dim
        self.max_period = max_period
        # Dummy parameter to detect if model is in fp16 mode
        self.dummy = nn.Parameter(
            torch.empty(0, dtype=torch.float), requires_grad=False
        )

    def forward(self, time):
        """
        Generates sinusoidal embeddings for the given timesteps.
        Args:
            time: Tensor of shape (batch_size,) containing timesteps
        Returns:
            Tensor of shape (batch_size, dim) containing the position embeddings
        """
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(self.max_period) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        embeddings = embeddings.to(self.dummy.dtype)
        return embeddings


class MainModel(BaseModel):
    """
    Main model class for the protein structure prediction using diffusion.
    Implements a diffusion-based approach to generate protein structures by learning
    the reverse process of a forward diffusion process.
    (SDE equation (1) and (2) in the paper)

    Input data dimensions:
    - single_repr: [batch_size, seq_length, 384] - Single residue features from Evoformer
    - pair_repr: [batch_size, seq_length, seq_length, 128] - Pairwise residue features
    - T: [batch_size, seq_length, 3] - Translation coordinates (x, y, z)
    - IR: [batch_size, seq_length, 3, 3] - Rotation matrices (SO(3))
    """
    def __init__(self, d_model=768, d_pair=256, n_layer=12, n_heads=32):
        """
        Args:
            d_model: Dimension of the model's hidden states after projecting Evoformer single-residual features (384 -> d_model)
            d_pair: Dimension for pairwise representations (128 -> d_pair)
            n_layer: Number of transformer layers
            n_heads: Number of attention heads
        """
        super(MainModel, self).__init__()

        self.init_diffusion_params()

        # Embeddings for diffusion timesteps
        # Input: [batch_size], Output: [batch_size, d_model]
        self.step_emb = SinusoidalPositionEmbeddings(dim=d_model) #Sinusoidal(...)
        
        # Projects single residue features to model dimension #h_i
        # Input: [batch_size, seq_length, 384]
        # Output: [batch_size, seq_length, d_model]
        self.x1d_proj = nn.Sequential(
            nn.LayerNorm(384), nn.Linear(384, d_model, bias=False)
        ) #Linear(LayerNorm(...))
        
        # Projects pair features to pair dimension #z_ij
        # Input: [batch_size, seq_length, seq_length, 128]
        # Output: [batch_size, seq_length, seq_length, d_pair]
        self.x2d_proj = nn.Sequential(
            nn.LayerNorm(128), nn.Linear(128, d_pair, bias=False)
        ) #LinearNoBias(LayerNorm(...))
        
        # Relative position encoding for pairs of residues 
        # Output: [seq_length, seq_length, d_pair]
        self.rp_proj = RelativePositionBias(
            num_buckets=64, max_distance=128, out_dim=d_pair
        ) #Embedding(Bucketize(...))

        # Main structure module that processes the protein features
        self.st_module = StructureModule(
            d_pair=d_pair,
            n_layer=n_layer,
            d_model=d_model,
            n_head=n_heads,
            dim_feedforward=1024,
            dropout=0.1,
        )

    def init_diffusion_params(self):
        """
        Initializes parameters for the diffusion process.
        Sets up the number of timesteps and noise schedules for translation and rotation.
        sigma ranges correspond to sigma_t in Eq. (A11) (Translation) and IGSO(3) variance (Rotation)
        """
        self.n_time_step = 500  # Total number of diffusion steps
        # Translation noise parameters
        self.tr_sigma_min = 0.1
        self.tr_sigma_max = 35
        # Rotation noise parameters
        self.rot_sigma_min = 0.02
        self.rot_sigma_max = 1.65
        # Time schedule for the diffusion process
        self.t_schedule = self._get_t_schedule(self.n_time_step)

    def forward_step(self, input_pose, mask, step, single_repr, pair_repr):
        """
        Performs sigma (score) for a single reverse-diffusion step.
        
        Args:
            input_pose: Tuple of (T, IR) where:
                - T: [batch_size, seq_length, 3] translation coordinates
                - IR: [batch_size, seq_length, 3, 3] rotation matrices
            mask: [batch_size, seq_length] boolean mask indicating valid positions
            step: [batch_size] current diffusion timesteps
            single_repr: [batch_size, seq_length, 384] single residue representations
            pair_repr: [batch_size, seq_length, seq_length, 128] pairwise residue representations
        
        Returns:
            Tuple (T_eps, IR_eps):
                - T_eps: [batch_size, seq_length, 3] predicted translation noise
                - IR_eps: [batch_size, seq_length, 3] predicted rotation noise
        """
        # Project single residue features and add timestep embeddings
        x1d = self.x1d_proj(single_repr) + self.step_emb(step)[:, None] #Linear(LayerNorm(h_i)) + Sinusoidal(...)
        # Project pair features
        x2d = self.x2d_proj(pair_repr) #LinearNoBias(LayerNorm(z_ij))
        T, IR = input_pose

        # Calculate relative positional encoding
        pos = torch.arange(T.shape[1], device=x1d.device)
        pos = pos.unsqueeze(1) - pos.unsqueeze(0)
        x2d = x2d + self.rp_proj(pos)[None] #LinearNoBias(LayerNorm(z_ij)) + Embedding(Bucketize(p_i))

        # Handle masking for valid positions
        z = (~mask).long().sum(-1, keepdims=True)
        mask = mask.masked_fill(z == 0, False)

        # Create attention bias mask
        bias = mask.float().masked_fill(mask, float("-inf"))[:, None, :, None]
        bias = bias.permute(0, 3, 1, 2)

        # Process through structure module
        T_eps, IR_eps = self.st_module((T, IR), x1d, x2d, bias) #for layers... to end for

        # Transform translation predictions to local frame
        T_eps = torch.matmul(IR.transpose(-1, -2), T_eps.unsqueeze(-1)).squeeze(-1)
        return T_eps, IR_eps

    def _gen_timestep(self, B, device):
        """
        Generates timesteps for batch training.
        Creates symmetric pairs of timesteps to improve training stability.
        
        Args:
            B: Batch size
            device: Device to put tensors on
        """
        # Generate half of the time steps, and the other half are the reverse
        time_step = torch.randint(self.n_time_step, size=(B // 2,)).to(device)
        time_step = torch.cat([time_step, self.n_time_step - 1 - time_step])
        return time_step

    def _get_t_schedule(self, n_time_step):
        """
        Creates a (linear) schedule of timesteps for the diffusion process.
        Linear in t; combined with the exponential schedule for translation and rotation.
        In _t_to_sigma, this yields a geometric noise schedule (large sigma first)
        """
        return torch.linspace(1, 0, n_time_step + 1)[:-1]

    def _t_to_sigma(self, time_step, device):
        """
        Converts timesteps to noise levels (sigmas) for both translation and rotation.
        Uses an exponential schedule for smooth transitions.
        """
        t = self.t_schedule[time_step].to(device)
        # Calculate translation sigma using exponential interpolation
        T_sigma = (self.tr_sigma_min ** (1 - t)) * (self.tr_sigma_max ** (t))
        # Calculate rotation sigma using exponential interpolation
        IR_sigma = (self.rot_sigma_min ** (1 - t)) * (self.rot_sigma_max ** (t))
        return T_sigma, IR_sigma

    def _gen_noise(self, time_step, T_size, IR_size, device):
        """
        Generates noise for both translation and rotation at given timesteps.
        Translation: iid N(0, sigma^2) matches Eq. (A20)
        Rotation: uses so3.batch_sample_vec, equivalent to IGSO(3) sampling in Eq. (A11)
        
        Args:
            time_step: [batch_size] Current diffusion timesteps
            T_size: Tuple (batch_size, seq_length) for translation tensor
            IR_size: Tuple (batch_size, seq_length) for rotation tensor
            device: Device to put tensors on
            
        Returns:
            Dictionary containing:
                - T_sigma: [batch_size] translation noise levels
                - IR_sigma: [batch_size] rotation noise levels
                - T_update: [batch_size, seq_length, 3] translation noise
                - T_score: [batch_size, seq_length, 3] translation scores (acutal scores that needs to be predicted)
                - so3_rot_update: [batch_size, seq_length, 3] rotation noise vectors
                - so3_rot_mat: [batch_size, seq_length, 3, 3] rotation noise matrices
                - so3_rot_score: [batch_size, seq_length, 3] rotation scores (actual scores that needs to be predicted)
                - so3_rot_score_norm: [batch_size, seq_length, 1] rotation score norms
        """
        T_sigma, IR_sigma = self._t_to_sigma(time_step, device)

        # Generate translation noise and scores
        T_update = torch.stack(
            [
                torch.normal(mean=0, std=T_sigma[i], size=(T_size[1], 3), device=device)
                for i in range(T_size[0])
            ],
            dim=0,
        )
        # Calculate translation score (negative noise normalized by variance)
        T_score = -T_update / T_sigma[..., None, None] ** 2

        # Helper function to generate rotation noise for a batch
        def gen_batch_sample(batch, rot_sigma, device):
            eps = rot_sigma.cpu().numpy()
            # Generate SO(3) rotation noise
            so3_rot_update_np = so3.batch_sample_vec(batch, eps=eps)
            so3_rot_update = torch.tensor(so3_rot_update_np, device=device)
            # Convert to rotation matrices
            so3_rot_mat = geometry.axis_angle_to_matrix(so3_rot_update.squeeze())
            # Calculate rotation scores
            so3_rot_score_np = so3.batch_score_vec(
                batch, vec=so3_rot_update_np, eps=eps
            )
            so3_rot_score = torch.tensor(so3_rot_score_np, device=device)
            so3_rot_score_norm = (
                so3.score_norm(torch.tensor([rot_sigma])).unsqueeze(-1).repeat(batch, 1)
            )

            return so3_rot_update, so3_rot_mat, so3_rot_score, so3_rot_score_norm

        # Generate rotation noise for each item in batch
        so3_rot_update_stack = []
        so3_rot_mat_stack = []
        so3_rot_score_stack = []
        so3_rot_score_norm_stack = []

        for b in range(IR_size[0]):
            L = IR_size[1]
            rot_sigma = IR_sigma[b]

            (
                so3_rot_update,
                so3_rot_mat,
                so3_rot_score,
                so3_rot_score_norm,
            ) = gen_batch_sample(L, rot_sigma, device)
            so3_rot_update_stack.append(so3_rot_update)
            so3_rot_mat_stack.append(so3_rot_mat)
            so3_rot_score_stack.append(so3_rot_score)
            so3_rot_score_norm_stack.append(so3_rot_score_norm)

        # Stack all rotation noise components
        so3_rot_update = torch.stack(so3_rot_update_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        so3_rot_mat = torch.stack(so3_rot_mat_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3, 3
        )
        so3_rot_score = torch.stack(so3_rot_score_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 3
        )
        rot_score_norm = torch.stack(so3_rot_score_norm_stack, dim=0).reshape(
            IR_size[0], IR_size[1], 1
        )

        return {
            "T_sigma": T_sigma,
            "IR_sigma": IR_sigma,
            "T_update": T_update,
            "T_score": T_score,
            "so3_rot_update": so3_rot_update,
            "so3_rot_mat": so3_rot_mat,
            "so3_rot_score": so3_rot_score,
            "so3_rot_score_norm": rot_score_norm,
        }

    def forward(self, data, compute_loss=True):
        """
        Forward pass of the model. Implements the training step for the diffusion model.
        
        Args:
            data: Dictionary containing:
                - single_repr: [batch_size, seq_length, 384] single residue representations
                - pair_repr: [batch_size, seq_length, seq_length, 128] pairwise residue representations
                - T: [batch_size, seq_length, 3] translation coordinates
                - IR: [batch_size, seq_length, 3, 3] rotation matrices
            compute_loss: Whether to compute loss (used during training)
            
        Returns:
            Dictionary containing:
                - loss: Scalar combined loss
                - T_diff_loss: Scalar translation loss
                - IR_diff_loss: Scalar rotation loss
                - update_loss: Same as loss
        """
        device = data["single_repr"].device
        B, L = data["single_repr"].shape[:2]
        T, IR = data["T"], data["IR"]
        # Create mask for valid positions (where coordinates are not NaN)
        mask = torch.isnan((IR.sum(-1) + T).sum(-1))

        # Generate timesteps and corresponding noise
        time_step = self._gen_timestep(B, device)
        noise_gen = self._gen_noise(time_step, T.size(), IR.size(), device)

        T_sigma, IR_sigma = noise_gen["T_sigma"], noise_gen["IR_sigma"]
        T_update = noise_gen["T_update"].type_as(T)
        T_score = noise_gen["T_score"].type_as(T)
        noise_gen["so3_rot_update"].type_as(IR)
        so3_rot_mat = noise_gen["so3_rot_mat"].type_as(IR)
        so3_rot_score = noise_gen["so3_rot_score"].type_as(IR)
        so3_rot_score_norm = noise_gen["so3_rot_score_norm"].type_as(IR)

        # Add noise to input coordinates
        T_perturbed = T + T_update
        IR_perturbed = torch.matmul(so3_rot_mat, IR)

        # Apply masking
        T_perturbed.masked_fill_(mask[..., None], 0.0)
        IR_perturbed.masked_fill_(mask[..., None, None], 0.0)
        
        # Get model predictions for noise
        pred_T_eps, pred_IR_eps = self.forward_step(
            (T_perturbed, IR_perturbed),
            mask,
            time_step,
            data["single_repr"],
            data["pair_repr"],
        )

        # Set up targets for loss computation
        target_T_eps = T_score
        target_IR_eps = so3_rot_score

        # Compute MSE losses for translation and rotation noise prediction
        T_diff_loss = (pred_T_eps - target_T_eps * T_sigma[..., None, None]) ** 2
        IR_diff_loss = (pred_IR_eps - target_IR_eps / so3_rot_score_norm) ** 2

        # Apply masking to losses
        T_diff_loss.masked_fill_(mask[..., None], 0)
        IR_diff_loss.masked_fill_(mask[..., None], 0)

        # Combine losses with equal unit weights
        loss = 1.0 * T_diff_loss.mean() + 1.0 * IR_diff_loss.mean()

        # Prepare output dictionary
        out = {}
        out["loss"] = loss
        out["T_diff_loss"] = T_diff_loss.mean()
        out["IR_diff_loss"] = IR_diff_loss.mean()
        out["update_loss"] = out["loss"]

        return out
