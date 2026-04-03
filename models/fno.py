import torch
from torch import nn
from neuralop.models import FNO
from neuralop.models import TFNO
import numpy as np
import inspect

class FluidSolidFNOmodel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.hidden_channels = config['model']['fno']['hidden_channels']
        self.n_layers = int(config['model']['fno']['layers'])
        
        # Use 1D modes
        n_modes_1d = config['model']['fno'].get('n_modes_1d', 64)
        domain_padding = config['model']['fno'].get('domain_padding', 0.1)

        # Input channels are now configurable for ablation experiments.
        self.available_features = {
            'w_old', 'b', 'r', 'diag_A', 'fi',
            'cluster_dist', 'sp_hf_0', 'sp_hf_1', 'sp_df_0', 'sp_df_1'
        }
        self.default_input_features = [
            'w_old', 'b', 'r', 'diag_A', 'fi',
            'cluster_dist', 'sp_hf_0', 'sp_hf_1', 'sp_df_0', 'sp_df_1'
        ]
        self.input_features = config['model'].get('input_features', self.default_input_features)
        if not isinstance(self.input_features, (list, tuple)) or len(self.input_features) == 0:
            raise ValueError("model.input_features must be a non-empty list")

        invalid_features = [f for f in self.input_features if f not in self.available_features]
        if invalid_features:
            raise ValueError(
                f"Unsupported input feature(s): {invalid_features}. "
                f"Available: {sorted(self.available_features)}"
            )
        
        self.in_channels = len(self.input_features)
        self.out_channels = 1

        film_cfg = config['model'].get('film', {})
        self.use_film = bool(film_cfg.get('enabled', config['model'].get('use_film', True)))
        self.physics_dim = int(film_cfg.get('physics_dim', 5))
        self.film_hidden_dim = int(film_cfg.get('hidden_dim', 32))

        fno_kwargs = {
            'n_modes': [n_modes_1d],
            'hidden_channels': self.hidden_channels,
            'in_channels': self.in_channels,
            'out_channels': self.out_channels,
            'n_layers': self.n_layers,
            'domain_padding': domain_padding,
            'positional_embedding': None, # 防止其在通道强塞网格坐标
            'use_channel_mlp': True,
            'norm': 'group_norm',
        }

        # Explicitly pass ratio parameters when configured so checkpoint loading is stable
        # across neuralop versions with different defaults.
        fno_signature = inspect.signature(FNO.__init__).parameters
        for ratio_key in ('lifting_channel_ratio', 'projection_channel_ratio'):
            ratio_val = config['model']['fno'].get(ratio_key)
            if ratio_val is not None and ratio_key in fno_signature:
                fno_kwargs[ratio_key] = float(ratio_val)

        self.fno = FNO(**fno_kwargs).to(config['training']['device'])

        if self.use_film:
            # FiLM network: maps physics params to gamma and beta for each FNO layer.
            # Output dim is 2 * n_layers * hidden_channels.
            self.physics_mlp = nn.Sequential(
                nn.Linear(self.physics_dim, self.film_hidden_dim),
                nn.GELU(),
                nn.Linear(self.film_hidden_dim, 2 * self.n_layers * self.hidden_channels)
            ).to(config['training']['device'])

            # Initialize MLP to output identity transformation initially (gamma=1, beta=0).
            nn.init.zeros_(self.physics_mlp[-1].weight)
            nn.init.zeros_(self.physics_mlp[-1].bias)
            self.physics_mlp[-1].bias.data[:self.n_layers * self.hidden_channels] = 1.0
        else:
            self.physics_mlp = None

        self.activation = nn.GELU()

    def forward(self, batch) -> torch.Tensor:
        # batch['A']: [B, N, N]
        # batch['b']: [B, N]
        # batch['w_old']: [B, N]
        # batch['fi']: [B, N]
        
        A = batch.get('A')
        b = batch['b']
        w_old = batch['w_old']
        fi = batch['fi']
        
        has_phys = 'phys_params' in batch and 'cluster_dist' in batch
        if has_phys:
            phys_params = batch['phys_params']     # [B, 5]
            cluster_dist = batch['cluster_dist']   # [B, N]
            sp_hf_0 = batch['sp_hf_0']             # [B, N]
            sp_hf_1 = batch['sp_hf_1']             # [B, N]
            sp_df_0 = batch['sp_df_0']             # [B, N]
            sp_df_1 = batch['sp_df_1']             # [B, N]
        else:
            phys_params = torch.zeros((w_old.size(0), self.physics_dim), device=w_old.device)
            cluster_dist = torch.zeros_like(w_old)
            sp_hf_0 = torch.full_like(w_old, 25.0)
            sp_hf_1 = torch.full_like(w_old, 25.0)
            sp_df_0 = torch.full_like(w_old, 2.0)
            sp_df_1 = torch.full_like(w_old, 2.0)

        # 1. Residual r
        if 'r' in batch:
            r = batch['r']
        else:
            if A is None:
                raise ValueError("Feature 'r' is required but neither batch['r'] nor batch['A'] is provided")
            Ax = torch.matmul(A, w_old.unsqueeze(-1)).squeeze(-1)
            r = b - Ax

        # 2. Diagonal of A
        if 'diag_A' in batch:
            diag_A = batch['diag_A']
        else:
            if A is None:
                raise ValueError("Feature 'diag_A' is required but neither batch['diag_A'] nor batch['A'] is provided")
            diag_A = torch.diagonal(A, dim1=1, dim2=2)

        # 3. Stack selected features: [B, C, N]
        feature_values = {
            'w_old': w_old,
            'b': b,
            'r': r,
            'diag_A': diag_A,
            'fi': fi,
            'cluster_dist': cluster_dist,
            'sp_hf_0': sp_hf_0,
            'sp_hf_1': sp_hf_1,
            'sp_df_0': sp_df_0,
            'sp_df_1': sp_df_1,
        }

        selected = []
        for feature_name in self.input_features:
            tensor = feature_values[feature_name]
            if tensor.ndim != 2:
                raise ValueError(
                    f"Feature '{feature_name}' must have shape [B, N], got {tuple(tensor.shape)}"
                )
            selected.append(tensor)
        x = torch.stack(selected, dim=1)
        
        film_params = None
        if self.use_film:
            film_params = self.physics_mlp(phys_params)
            film_params = film_params.view(-1, self.n_layers, 2, self.hidden_channels)

        # Break down neuralop FNO forward to inject FiLM
        x = self.fno.lifting(x)
        if self.fno.domain_padding is not None:
            x = self.fno.domain_padding.pad(x)

        for layer_idx, (fno_block, fno_skip) in enumerate(zip(self.fno.fno_blocks.convs, self.fno.fno_blocks.fno_skips)):
            skip = fno_skip(x)
            x = fno_block(x) + skip

            if self.use_film:
                gamma = film_params[:, layer_idx, 0, :].unsqueeze(-1)
                beta  = film_params[:, layer_idx, 1, :].unsqueeze(-1)
                x = x * gamma + beta
            
            if layer_idx < self.n_layers - 1:
                x = self.activation(x)

        if self.fno.domain_padding is not None:
            x = self.fno.domain_padding.unpad(x)
            
        x = self.fno.projection(x) # Output: [B, 1, N]
        
        return x.squeeze(1) # [B, N]

    def name(self) -> str:
        return "FluidSolidFNOmodel_1D"