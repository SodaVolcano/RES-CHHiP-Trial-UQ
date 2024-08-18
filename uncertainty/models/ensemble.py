"""
An ensemble of models
"""

from typing import Callable
from torch import nn, vmap
import torch
from torch.func import stack_module_state, functional_call  # type: ignore


class DeepEnsemble(nn.Module):
    """
    An ensemble of models, produces multiple outputs from each member
    """

    def __init__(
        self,
        model: Callable[..., nn.Module] | list[nn.Module],
        ensemble_size: int,
        *args,
        **kwargs,
    ):
        """
        An ensemble of models

        Parameters
        ----------
        ensemble_size : int
            The number of models in the ensemble
        model : Callable[..., nn.Module] | list[nn.Module]
            A callable PyTorch model class, or a list of PyTorch models
        *args, **kwargs : list, dict
            Additional arguments to pass to the model_fn
        """
        super().__init__()
        self.models = (
            [model(*args, **kwargs) for _ in range(ensemble_size)]
            if isinstance(model, Callable)
            else model
        )
        self.ensemble_params, self.ensemble_buffers = stack_module_state(self.models)

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        def call_model(params: dict, buffers: dict, x: torch.Tensor) -> torch.Tensor:
            return functional_call(self.models[0], (params, buffers), (x,))

        # each model should have different randomness (dropout)
        return vmap(call_model, in_dims=(0, 0, None), randomness="different")(
            self.ensemble_params, self.ensemble_buffers, x
        )


#
class BatchEnsembleConv3D(nn.Module):
    """
    Convolution 3D using an ensemble of estimators with a shared weight matrix

    Taken from https://github.com/ENSTA-U2IS-AI/torch-uncertainty/blob/main/torch_uncertainty/layers/batch_ensemble.py

    Caveats
    -------
    In training when passing a mini-batch of B examples, the mini-batch is
    partitioned into smaller batches of `B // n_estimators` examples and
    given to each member. The remaining examples are discarded. In
    testing, you should duplicate the test batch `n_estimators` times.

    Explanation
    -----------
    Each ensemble member is represented using an `r` and `s` weight vectors
    ("fast weights"), where `r` have same shape as the input channels and `s`
    have same shape as the output channels. The full weight matrix `W_i` for
    the `i`-th member as derived as
        `W_i = W * r_i * s_i^T`
    where `W` is the shared weight (the Conv3D weights), `*` denotes the
    Hadamard product, and `^T` denotes the transpose.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int, int],
        n_estimators: int,
        stride: int | tuple[int, int, int] = 1,
        padding: str | int | tuple[int, int, int] = 0,
        dilation: int | tuple[int, int, int] = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ):
        """
        Initialises a 3D convolutional layer for a batch ensemble

        Parameters
        ----------
        in_channels : int
            Number of input channels for the 3D convolution.
        out_channels : int
            Number of output channels for the 3D convolution.
        kernel_size : int or tuple of int
            Size of the convolutional kernel. Can be an int or a tuple of 3 ints
            representing the size in each spatial dimension.
        n_estimators : int
            Number of estimators in the ensemble. Determines the number of
            r and s weight groups.
        stride : int or tuple of int, default=1
            Stride of the convolution. Can be an int or a tuple of 3 ints
            representing the stride in each spatial dimension.
        padding : str, int, or tuple of int, default=0
            Padding added to all three spatial dimensions. Can be 'same',
            an int, or a tuple of 3 ints.
        dilation : int or tuple of int, default=1
            Dilation rate for dilated convolutions. Can be an int or a tuple of
            3 ints representing the dilation in each spatial dimension.
        bias : bool, default=True
            If True, adds a learnable bias for each estimator. If False, no
            bias is added.
        device : torch.device, optional
            The device on which to allocate tensors. If None, defaults to the
            current device.
        dtype : torch.dtype, optional
            The data type for the tensors. If None, defaults to the current
            default dtype.

        Attributes
        ----------
        conv : nn.Conv3d
            3D convolutional layer with the specified parameters.
        r_weights : nn.Parameter
            Learnable weight vectors for each estimator corresponding to the input channels.
        s_weights : nn.Parameter
            Learnable weight vectors for each estimator corresponding to the output channels.
        bias : nn.Parameter or None
            Optional learnable bias for each estimator, if `bias=True`. Otherwise,
            it is set to None.
        """
        super().__init__()

        self.n_estimators = n_estimators

        self.conv = nn.Conv3d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False,
            device=device,
        )

        # Fast weights r and s for each ensemble member
        self.r_weights = nn.Parameter(torch.empty((n_estimators, in_channels)))
        self.s_weights = nn.Parameter(torch.empty((n_estimators, out_channels)))

        if bias:
            self.bias = nn.Parameter(torch.empty((n_estimators, out_channels)))
        else:
            self.register_parameter("bias", None)

    def init_weights(self, init_fn=None):
        """
        Initialise the model's weights using the provided initialiser function.

        Parameters
        ----------
        init_fn : callable, optional
            A function to initialise the weights. If None, uses Xavier uniform initialisation.
        """
        init_fn = init_fn or nn.init.xavier_uniform_

        init_fn(self.r_weights)
        init_fn(self.s_weights)
        if self.bias is not None:
            init_fn(self.bias)

    def get_fast_weight_matrix(
        self, weights: torch.Tensor, examples_per_estimator: int
    ) -> torch.Tensor:
        """
        Stack fast weight vectors into a matrix.

        This function stacks the fast weight vectors from multiple estimators into a single matrix. 
        For each estimator, the weight vectors are vertically repeated `N` times, where `N` is the 
        number of examples per estimator. These repeated vectors are then stacked horizontally to 
        form a matrix of shape `(N * M, C)`.

        Examples
        --------
        For 3 estimators, 2 examples per estimator, and 5 channels, the fast weights are:

        .. math::
            \text{Fast weight for i-th estimator} = [w_{i1}, w_{i2}, w_{i3}, w_{i4}, w_{i5}]
        
        The resulting matrix will be:

        .. math::
            \begin{bmatrix}
            w_{11} & w_{12} & w_{13} & w_{14} & w_{15} \\
            w_{11} & w_{12} & w_{13} & w_{14} & w_{15} \\
            w_{21} & w_{22} & w_{23} & w_{24} & w_{25} \\
            w_{21} & w_{22} & w_{23} & w_{24} & w_{25} \\
            w_{31} & w_{32} & w_{33} & w_{34} & w_{35} \\
            w_{31} & w_{32} & w_{33} & w_{34} & w_{35}
        \end{bmatrix}
    """
        repeats_per_estimator = torch.full(
            (self.n_estimators,), examples_per_estimator, device=weights.device
        )
        # unsqueeze to add extra dimensions to match 3D input
        return (
            torch.repeat_interleave(weights, repeats_per_estimator, dim=0)
            .unsqueeze(-1)
            .unsqueeze(-1)
            .unsqueeze(-1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        examples_per_estimator = batch_size // self.n_estimators

        r_matrix = self.get_fast_weight_matrix(self.r_weights, examples_per_estimator)
        s_matrix = self.get_fast_weight_matrix(self.s_weights, examples_per_estimator)
        bias = (
            None
            if self.bias is None
            else self.get_fast_weight_matrix(self.bias, examples_per_estimator)
        )
        # vectorised form, output = ((X * R)W) * S + bias where X is minibatch matrix
        return self.conv(x * r_matrix) * s_matrix + (bias if bias is not None else 0)
