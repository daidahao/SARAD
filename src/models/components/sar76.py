import torch
from torch import nn, Tensor
import timeit
import rootutils
rootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)

from src import utils
from src.models.components.sar73 import SAR73

from src.utils.logging_utils import log

class SAR76(SAR73):
    def __init__(self, input_size: int, window_size: int, \
        # main hyperparameters
        model_size: int, num_layers: int, num_heads: int, \
        # progression hyperparameters
        num_patches: int, \
        # detector hyperparameters
        detector_size: int,
        dropout: float, \
        is_diagnoal_masked: bool = False,
    ):
        assert num_patches == 2, 'Number of patches must be 2.'
        super(SAR76, self).__init__(input_size, window_size, 
            model_size, num_layers, num_heads, 
            num_patches, 
            dropout,
            is_diagnoal_masked,
        )

        self.detector = nn.Sequential(
            nn.Linear(num_heads*input_size, detector_size), nn.ReLU(),
            nn.Linear(detector_size, num_heads*input_size), nn.ReLU(),
        )
        
        self.register_buffer('detec_avg', torch.tensor(0.))
        self.register_buffer('detec_std', torch.tensor(1.))
        self.register_buffer('recon_avg', torch.tensor(0.))
        self.register_buffer('recon_std', torch.tensor(1.))

        self.register_buffer('rdiag_avg', torch.zeros(input_size))
        self.register_buffer('rdiag_std', torch.ones(input_size))

        self.register_buffer('ddiag_avg', torch.zeros(input_size))
        self.register_buffer('ddiag_std', torch.ones(input_size))
    
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

        # compute q: mean reduction of s
        s_detached = s_all.detach()
        s_reduced = s_detached[:, :, 0]-s_detached[:, :, 1]
        s_reduced = torch.nn.functional.relu(s_reduced)
        # q shape: [batch_size, n_heads, input_size]
        q = s_reduced[:, -1].sum(-2)
        # q *= self.input_size

        # compute q_bar: reconstructed q
        q_bar = self.detector(q.flatten(1))
        q_bar = q_bar.reshape_as(q)

        return x_hat, s_all, q, q_bar


if __name__ == '__main__':
    batch_size, window_size, input_size = 32, 100, 25
    model_size, num_layers, num_heads = 128, 3, 8
    num_patches = 2
    svdd_c_size = 8
    dropout = 0.1

    model = SAR76(input_size, window_size, \
        model_size, num_layers, num_heads, \
        num_patches, \
        dropout, \
    )
    x = torch.randn(batch_size, window_size, input_size)
    
    start = timeit.default_timer()
    for _ in range(100):
        x_hat, s = model(x)
    stop = timeit.default_timer()

    print('Time: ', stop - start)  
    print(x.shape, x_hat.shape, s.shape)
