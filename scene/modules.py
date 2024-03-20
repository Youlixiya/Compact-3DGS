import math

from einops import rearrange, repeat
from jaxtyping import Float
import torch
from torch import nn, Tensor
import torch.nn.functional as F


class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        """
        Initialize the RMSNorm normalization layer.

        Args:
            dim (int): The dimension of the input tensor.
            eps (float, optional): A small value added to the denominator for numerical stability. Default is 1e-6.

        Attributes:
            eps (float): A small value added to the denominator for numerical stability.
            weight (nn.Parameter): Learnable scaling parameter.

        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        """
        Apply the RMSNorm normalization to the input tensor.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The normalized tensor.

        """
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        """
        Forward pass through the RMSNorm layer.

        Args:
            x (torch.Tensor): The input tensor.

        Returns:
            torch.Tensor: The output tensor after applying RMSNorm.

        """
        output = self._norm(x.float()).type_as(x)
        return output * self.weight

class CrossAttentionBlock(nn.Module):
    """
    Transformer block that takes in a cross-attention condition.
    Designed for SparseLRM architecture.
    """
    # Block contains a cross-attention layer, a self-attention layer, and an MLP
    def __init__(self, inner_dim: int, cond_dim: int, num_heads: int, eps: float,
                 attn_drop: float = 0., attn_bias: bool = False,
                 mlp_ratio: float = 4., mlp_drop: float = 0.):
        super().__init__()
        self.norm1 = RMSNorm(inner_dim, eps=eps)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads, kdim=cond_dim, vdim=cond_dim,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm2 = RMSNorm(inner_dim, eps=eps)
        self.self_attn = nn.MultiheadAttention(
            embed_dim=inner_dim, num_heads=num_heads,
            dropout=attn_drop, bias=attn_bias, batch_first=True)
        self.norm3 = RMSNorm(inner_dim, eps=eps)
        self.mlp = nn.Sequential(
            nn.Linear(inner_dim, int(inner_dim * mlp_ratio)),
            nn.SiLU(),
            nn.Dropout(mlp_drop),
            nn.Linear(int(inner_dim * mlp_ratio), inner_dim),
            nn.Dropout(mlp_drop),
        )

    def forward(self, hidden_states, encoder_hidden_states):
        # x: [N, L, D]
        # cond: [N, L_cond, D_cond]
        hidden_states = hidden_states + self.cross_attn(self.norm1(hidden_states), encoder_hidden_states, encoder_hidden_states, need_weights=False)[0]
        before_sa = self.norm2(hidden_states)
        hidden_states = hidden_states + self.self_attn(before_sa, before_sa, before_sa, need_weights=False)[0]
        hidden_states = hidden_states + self.mlp(self.norm3(hidden_states))
        return hidden_states
        
class TriplaneTokens(nn.Module):
    def __init__(self,
                 resolution: int = 32,
                 num_components: int = 1024) -> None:
        super().__init__()
        self.resolution = resolution
        self.num_components = num_components
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, self.num_components, self.resolution, self.resolution),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(self.num_components)
            )

    def forward(self, batch_size: int) -> torch.Tensor:
        return rearrange(
            repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size),
            "B Np Ct Hp Wp -> B (Np Hp Wp) Ct",
        )

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        # print(tokens.shape)
        batch_size, Nt, Ct = tokens.shape
        assert Nt == self.resolution**2 * 3
        assert Ct == self.num_components
        return rearrange(
            tokens,
            "B (Np Hp Wp) Ct -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.resolution,
            Wp=self.resolution,
        )

class Transformer(nn.Module):
    def __init__(self,
                 in_channels = 1024,
                 num_attention_heads = 16,
                 attention_head_dim = 32,
                 cond_dim = 512,
                 num_layers = 4,
                #  num_groups = 32,
                 dropout = 0.0,
                 eps = 1e-6):
        super().__init__()
        self.in_channels = in_channels
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.num_layers = num_layers
        # self.num_groups = num_groups
        inner_dim = self.num_attention_heads * self.attention_head_dim
        # self.norm = torch.nn.GroupNorm(
        #     num_groups=self.num_groups,
        #     num_channels=self.in_channels,
        #     eps=1e-6,
        #     affine=True,
        # )
        self.norm = RMSNorm(self.in_channels, eps)
        self.proj_in = nn.Linear(self.in_channels, inner_dim)
        self.transformer_blocks = nn.ModuleList(
            [
                CrossAttentionBlock(
                    inner_dim,
                    cond_dim,
                    self.num_attention_heads,
                    eps,
                    attn_bias=False,
                    mlp_ratio=4,
                    mlp_drop=dropout
                )
                for d in range(num_layers)
            ]
        )
        self.proj_out = nn.Linear(inner_dim, self.in_channels)
    
    def forward(self, hidden_states, encoder_hidden_states):
        # batch, seq_len, hidden_dim = hidden_states.shape
        residual = hidden_states

        hidden_states = self.norm(hidden_states)
        hidden_states = self.proj_in(hidden_states)
        for block in self.transformer_blocks:
            hidden_states = block(
                hidden_states,
                encoder_hidden_states
            )
        hidden_states = self.proj_out(hidden_states)
        output = hidden_states + residual
        return output
        
        

def triplane_sample(plane_coef, in_tensor):
        """Sample features from this encoder. Expects in_tensor to be in range [0, resolution]"""
        num_components = plane_coef.shape[1]

        original_shape = in_tensor.shape
        in_tensor = in_tensor.reshape(-1, 3)

        plane_coord = torch.stack([in_tensor[..., [0, 1]], in_tensor[..., [0, 2]], in_tensor[..., [1, 2]]], dim=0)

        # Stop gradients from going to sampler
        plane_coord = plane_coord.detach().view(3, -1, 1, 2)
        plane_features = F.grid_sample(
            plane_coef, plane_coord, align_corners=True
        )  # [3, num_components, flattened_bs, 1]
        plane_features = plane_features.sum(0).squeeze(-1).T

        return plane_features.reshape(*original_shape[:-1], num_components)