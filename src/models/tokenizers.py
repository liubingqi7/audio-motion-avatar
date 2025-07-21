import math
import torch
import torch.nn as nn
from einops import rearrange, repeat
from typing import Optional

class TriplaneLearnablePositionalEmbedding(nn.Module):
    def __init__(self, num_channels: int = 1024, plane_size: int = 32):
        super().__init__()
        self.plane_size = plane_size
        self.num_channels = num_channels
        self.embeddings = nn.Parameter(
            torch.randn(
                (3, num_channels, plane_size, plane_size),
                dtype=torch.float32,
            )
            * 1
            / math.sqrt(num_channels)
        )

    def forward(self, batch_size: int, cond_embeddings: Optional[torch.Tensor] = None) -> torch.Tensor:
        embeddings = repeat(self.embeddings, "Np Ct Hp Wp -> B Np Ct Hp Wp", B=batch_size)
        if cond_embeddings is not None:
            embeddings = embeddings + cond_embeddings
        return rearrange(
            embeddings,
            "B Np Ct Hp Wp -> B Ct (Np Hp Wp)",
        )

    def detokenize(self, tokens: torch.Tensor) -> torch.Tensor:
        batch_size, Ct, Nt = tokens.shape
        assert Nt == self.plane_size**2 * 3
        assert Ct == self.num_channels
        return rearrange(
            tokens,
            "B Ct (Np Hp Wp) -> B Np Ct Hp Wp",
            Np=3,
            Hp=self.plane_size,
            Wp=self.plane_size,
        )
