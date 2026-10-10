"""Run a temporal-spatial CNN and bidirectional GRU regressor.

Input is a float tensor (batch, channels, timepoints). Output is
(batch, outputs). The caller defines channel count, duration, and targets.
The layer topology follows TemporalSpatialCNNRNN in brain_decoding's
models/torch_architectures/cnn.py, with its documented default parameters.
"""
from __future__ import annotations

import torch
from torch import nn

FILTERS = 512
FILTER_LENGTH = 4
CONV_STRIDE = 3
POOL_LENGTH = 3
POOL_STRIDE = 2
CONV_BLOCKS = 3
GRU_WIDTH = 64
DROPOUT = 0.3


class TemporalSpatialCNNGRU(nn.Module):
    """Temporal-spatial convolution and bidirectional GRU regressor.

    Parameters
    ----------
    channels : int
        Number of input channels.
    input_length : int
        Number of samples per input window.
    outputs : int
        Number of regression or classification outputs.
    n_convolution_layers : int, optional
        Number of convolution blocks. Default is 3.
    conv_stride : int, optional
        Temporal convolution stride. Default is 3.
    drop_out : float, optional
        Block dropout probability. Default is 0.3.
    """

    def __init__(
        self, input_channels: int, input_length: int, outputs: int,
        n_convolution_layers: int = CONV_BLOCKS,
        conv_stride: int = CONV_STRIDE,
        drop_out: float = DROPOUT,
    ) -> None:
        super().__init__()
        if min(input_channels, input_length, outputs) < 1:
            raise ValueError(
                "Input channels, length, and outputs must be positive."
            )
        if type(n_convolution_layers) is not int or (
            n_convolution_layers < 1
        ):
            raise ValueError(
                "n_convolution_layers must be a positive integer."
            )
        if type(conv_stride) is not int or conv_stride < 1:
            raise ValueError("conv_stride must be a positive integer.")
        if not 0 <= drop_out < 1:
            raise ValueError("drop_out must be in [0, 1).")
        self.channels = input_channels
        self.input_length = input_length
        self.conv_timespat = nn.Conv2d(
            1, FILTERS, (FILTER_LENGTH, input_channels),
            stride=(conv_stride, 1),
        )
        self.bnorm = nn.BatchNorm2d(FILTERS, affine=True, eps=1e-5)
        self.activation = nn.ELU()
        self.pool = nn.MaxPool2d(
            kernel_size=(POOL_LENGTH, 1),
            stride=(POOL_STRIDE, 1),
        )
        blocks: list[nn.Module] = []
        for index in range(n_convolution_layers):
            blocks.extend((
                nn.Dropout(p=drop_out),
                nn.Conv2d(
                    FILTERS, FILTERS, (FILTER_LENGTH, 1),
                    stride=(conv_stride, 1),
                    padding=(
                        ((FILTER_LENGTH - 1) * conv_stride) // 2,
                        0,
                    ),
                ),
            ))
            if index:
                blocks.extend((
                    nn.BatchNorm2d(FILTERS, momentum=0.1,
                                   affine=True, eps=1e-5),
                    nn.ELU(),
                    nn.MaxPool2d(
                        kernel_size=(POOL_LENGTH, 1),
                        stride=(POOL_STRIDE, 1),
                    ),
                ))
        self.conv_pool_block = nn.Sequential(*blocks)
        self.gru = nn.GRU(
            FILTERS, GRU_WIDTH, 1, batch_first=True,
            bidirectional=True,
        )
        self.output_projection = nn.Linear(2 * GRU_WIDTH, outputs)
        self.apply(self._initialise_weights)

    @staticmethod
    def _initialise_weights(module: nn.Module) -> None:
        """Apply the cited architecture's convolution/linear initialization."""
        if isinstance(module, (nn.Conv2d, nn.Linear)):
            nn.init.kaiming_normal_(
                module.weight, mode="fan_in",
                nonlinearity="leaky_relu",
            )
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Predict from a (batch, channels, timepoints) tensor."""
        if x.ndim != 3 or x.shape[1:] != (
            self.channels, self.input_length,
        ):
            raise ValueError(
                "Expected input shape (batch, "
                f"{self.channels}, {self.input_length}), got {tuple(x.shape)}."
            )
        x = x.transpose(1, 2).unsqueeze(1)
        x = self.pool(self.activation(self.bnorm(self.conv_timespat(x))))
        x = self.conv_pool_block(x)
        if x.shape[-1] != 1 or x.shape[-2] < 1:
            raise ValueError(f"Unexpected CNN output shape {tuple(x.shape)}.")
        x = x.squeeze(-1).transpose(1, 2)
        x = self.gru(x)[0][:, -1]
        return self.output_projection(x)
