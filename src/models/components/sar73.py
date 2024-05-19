import math
from typing import Optional
import torch
from torch import nn, Tensor
import einops
import timeit

# Source: https://pytorch.org/tutorials/beginner/transformer_tutorial.html
class SpatialEncoding(nn.Module):
    def __init__(self, input_size: int, model_size: int, \
                 requires_grad: bool = True):
        super().__init__()
        self.se = nn.Parameter(
            torch.randn(1, 1, input_size, model_size), \
                requires_grad=requires_grad)

    def forward(self) -> Tensor:
        return self.se

class Embedding(nn.Module):
    def __init__(self, input_size:int, 
        patch_size: int, model_size: int, \
        dropout: float):
        super().__init__()
        self.encoding = nn.Linear(patch_size, model_size)
        self.spatial_encoding = SpatialEncoding(input_size, model_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, num_patches, input_size, patch_size]``
        """
        # patching
        # shape: [batch_size, n_patches, input_size, model_size]
        x = self.encoding(x) + self.spatial_encoding()
        return self.dropout(x)


class Patching(nn.Module):
    def __init__(self, num_patches: int):
        super().__init__()
        self.num_patches = num_patches
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, window_size, input_size]``
        """
        # shape: [batch_size, n_patches, input_size, patch_size]
        x = einops.rearrange(x, 'b (p s) i -> b p i s', p=self.num_patches)
        return x

class Unpatching(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, num_patches, input_size, patch_size]``
        """
        # shape: [batch_size, window_size, input_size]
        x = einops.rearrange(x, 'b p i s -> b (p s) i')
        return x

class Attention(nn.Module):
    def __init__(self, input_size: int, model_size: int, \
        n_heads: int, dropout: float, bias: bool, \
        is_diagnoal_masked: bool, \
    ):
        super().__init__()
        self.model_size = model_size
        self.n_heads = n_heads
        self.head_size = model_size // n_heads
        assert model_size % n_heads == 0, 'model_size must be divisible by n_heads'

        self.Q = nn.Linear(self.model_size, self.model_size, bias)
        self.K = nn.Linear(self.model_size, self.model_size, bias)
        self.V = nn.Linear(self.model_size, self.model_size, bias)

        self.linear = nn.Linear(self.model_size, self.model_size)

        self.dropout = nn.Dropout(dropout)

        self.is_diagnoal_masked = is_diagnoal_masked

        diag_mask = 1.0 - torch.eye(input_size, input_size).\
            unsqueeze(0).unsqueeze(0)
        self.register_buffer('diag_mask', diag_mask)


    
    def forward(self, q: Tensor, k: Tensor, v: Tensor, \
                s: Optional[Tensor] = None) -> Tensor:
        """
        Arguments:
            q, k, v: Tensor, shape ``[batch_size, input_size, model_size]``
            s: Tensor, shape ``[batch_size, n_heads, input_size, input_size]``
        """
        batch_size, input_size, _ = q.size()

        # shape: [batch_size, input_size, n_heads, head_size]
        v = self.V(v).view(batch_size, input_size, self.n_heads, self.head_size)

        # Self-attention
        # shape: [batch_size, n_heads, input_size, input_size]
        if s is None:
            q = self.Q(q).view(batch_size, input_size, self.n_heads, self.head_size)
            k = self.K(k).view(batch_size, input_size, self.n_heads, self.head_size)
            scores1 = torch.einsum("bqhe,bkhe->bhqk", [q, k]) / math.sqrt(self.head_size)
            s = torch.softmax(scores1, dim=-1)
            if self.is_diagnoal_masked:
                s = s * self.diag_mask
                s = s / (s.sum(dim=-1, keepdim=True)+1e-6)
        else:
            if self.is_diagnoal_masked:
                s = s * self.diag_mask
            s = s - s.min(dim=1, keepdim=True)[0]
            s = s / (s.sum(dim=-1, keepdim=True)+1e-6)
        s_d = self.dropout(s)

        # attention scores
        # shape: [batch_size, input_size, n_heads, head_size]
        attention = torch.einsum("bhql,blhd->bqhd", [s_d, v])
        # shape: [batch_size, input_size, model_size]
        attention = attention.reshape(batch_size, input_size, self.model_size)
        # todo: remove linear projection
        attention = self.linear(attention)
        return attention, s

class SpatialEncoder(nn.Module):
    def __init__(self, input_size: int, \
        model_size: int, feedforward_size: int, \
        num_heads: int, dropout: float, \
        bias: bool, is_diagnoal_masked: bool, \
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attention = Attention(input_size, model_size, \
                                   num_heads, dropout, \
                                   bias, is_diagnoal_masked)
        self.norm1 = nn.LayerNorm(model_size)
        self.norm2 = nn.LayerNorm(model_size)
        self.dropout = nn.Dropout(dropout)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_size, feedforward_size, bias), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(feedforward_size, model_size, bias), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, s: Optional[Tensor] = None) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, input_size, model_size]``
            s: Tensor, shape ``[batch_size, n_heads, input_size, input_size]``
        """
        if x.dim() == 4:
            batch_size, num_patches, input_size, model_size = x.size()
            z = x.view(batch_size*num_patches, input_size, model_size)
        else:
            z = x
        attention, s = self.attention(z, z, z, s)
        z = self.norm1(z + self.dropout(attention))
        z = self.norm2(z + self.feed_forward(z))
        if x.dim() == 4:
            z = z.view(batch_size, num_patches, input_size, model_size)
            s = s.view(batch_size, num_patches, self.num_heads, input_size, input_size)
        return z, s

class Decoder(nn.Module):
    def __init__(self, \
        patch_size: int, model_size: int
    ):
        super().__init__()
        self.ln = nn.LayerNorm(model_size)
        self.linear = nn.Linear(model_size, patch_size)

    def forward(self, x: Tensor) -> Tensor:
        """
        Arguments:
            x: Tensor, shape ``[batch_size, n_patches, input_size, model_size]``
        """
        # layer norm
        x = self.ln(x)
        # linear decoding
        # shape: [batch_size, n_patches, input_size, patch_size]
        x = self.linear(x)
        return x


class SAR73(nn.Module):
    def __init__(self, input_size: int, window_size: int, \
        # main hyperparameters
        model_size: int, num_layers: int, num_heads: int, \
        # progression hyperparameters
        num_patches: int, \
        dropout: float, \
        is_diagnoal_masked: bool = True, \
    ):
        """Initialize an `SAR73` module.
        
        :param input_size: The number of input features.
        :param window_size: The window size.
        :param model_size: The model size.
        :param feedforward_size: The feedforward size.
        :param num_layers: The number of layers.
        :param num_heads: The number of heads.
        :param dropout: The dropout rate.
        """

        super(SAR73, self).__init__()

        feedforward_size = 4 * model_size

        patch_size = window_size // num_patches

        self.patching = Patching(num_patches)
        
        self.embedding = Embedding(input_size, \
            patch_size, model_size, \
            dropout)
        
        self.encoders = nn.ModuleList([
            SpatialEncoder(input_size, model_size, feedforward_size,
                num_heads, dropout, \
                bias=True, is_diagnoal_masked=is_diagnoal_masked) 
            for _ in range(num_layers)
        ])

        self.decoder = Decoder(patch_size, model_size)

        self.unpatching = Unpatching()



    
    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        """
        :param x: The input tensor, shape ``[batch_size, window_size, input_size]``
        :return x_hat: The reconstructed x tensor, same shape as `x`.
        :return s: The spatial association tensor list.
        :return s_add: The spatial association tensor list.
        :return s_sub: The spatial association tensor list.
        """

        # # normalization (Non-stationary Transformer)
        # x_mean = x.mean(1, keepdim=True)
        # x_std = torch.sqrt(
        #     torch.var(x, 1, keepdim=True, unbiased=False) + 1e-5)
        # x = (x - x_mean) / x_std

        # patch series
        # shape: [batch_size, n_patches, input_size, patch_size]
        x = self.patching(x)
        
        # embed series
        # z shape: [batch_size, n_patches, input_size, model_size]
        z = self.embedding(x)

        # encode
        s_all = []
        for encoder in self.encoders:
            # s_layer shape: [batch_size, n_patches, n_heads, input_size, input_size]
            z, s_layer = encoder(z)
            s_all.append(s_layer)
        # shape: [batch_size, n_layers, n_patches, n_heads, input_size, input_size]
        s_all = torch.stack(s_all, dim=1)

        # decode
        # shape: [batch_size, n_patches, input_size, patch_size]
        x_hat = self.decoder(z)

        # unpatch
        # shape: [batch_size, window_size, input_size]
        x_hat = self.unpatching(x_hat)

        # # de-normalization
        # x_hat = x_hat * x_mean + x_std

        return x_hat, s_all


if __name__ == '__main__':
    batch_size, window_size, input_size = 32, 100, 25
    model_size, feedforward_size = 128, 128
    num_layers, num_heads = 3, 8
    num_patches = 2
    dropout = 0.1

    model = SAR73(input_size, window_size, \
                  model_size, feedforward_size, \
                    num_layers, num_heads, \
                    num_patches, \
                    dropout)
    x = torch.randn(batch_size, window_size, input_size)
    
    start = timeit.default_timer()
    for _ in range(100):
        z, s = model(x)
    stop = timeit.default_timer()

    print('Time: ', stop - start)  
    print(x.shape, z.shape, s.shape)
