# Copyright 2022 Vittorio Mazzia. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from typing import Callable, Tuple

import torch
from torch.nn import Linear, Conv2d, MaxPool2d

class FFLinear(Linear):
    """Fully connected layer Forward-Forward compatible
    """

    def __init__(self, in_features: int, out_features: int, activation: torch.nn,
                 optimizer: torch.optim, layer_optim_learning_rate: float, threshold: float, loss_fn: Callable,
                 method: str = "MSE", bias: bool = True):
        """Initialize layer

        Args:
            in_features (int): input features
            out_features (int): output features
            activation (torch.nn): layer activation
            optimizer (torch.optim): layer optimizer
            layer_optim_learning_rate (float): learning rate
            threshold (float): loss function threshold. TODO: Future implementaton should not have it here. Create class loss
            loss_fn (Callable): layer level loss function
            bias (bool, optional): if biase. Defaults to True.
        """
        super(FFLinear, self).__init__(in_features, out_features, bias)

        self.activation = activation
        self.optimizer = optimizer(self.parameters(), lr=layer_optim_learning_rate)
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        if (len(x.shape) > 2):
            x = x.view(x.size(0), -1)

        x = x / (x.norm(2, 1, keepdim=True) + 1e-8)  # mormalize input
        return self.activation(
            torch.mm(x, self.weight.T) +
            self.bias.unsqueeze(0))

    def train_layer(self, X_pos: torch.Tensor, X_neg: torch.Tensor, before: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Train layer with FF algorithm

        Args:
            X_pos (torch.Tensor): batch of positive examples
            X_neg (torch.Tensor): batch of negative examples
            before (bool): if True, successive layers get previous layers output before update

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: batch of positive and negative predictions and loss value
        """
        X_pos_out = self.forward(X_pos)
        X_neg_out = self.forward(X_neg)

        loss = self.loss_fn(X_pos_out, X_neg_out, self.threshold, self.method)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        if before:
            return X_pos_out.detach(), X_neg_out.detach(), loss.item()
        else:
            return self.forward(X_pos).detach(), self.forward(X_neg).detach(), loss.item()


class FFConv2d(Conv2d):
    """Convolutional layer Forward-Forward compatible
    """

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int,
                 activation: torch.nn, optimizer: torch.optim, layer_optim_learning_rate: float,
                 threshold: float, loss_fn: Callable, method: str = "MSE", bias: bool = True):
        """Initialize layer

        Args:
            in_channels (int): number of input channels
            out_channels (int): number of output channels
            kernel_size (int): size of the convolutional kernel
            stride (int): stride of the convolution
            activation (torch.nn): layer activation
            optimizer (torch.optim): layer optimizer
            layer_optim_learning_rate (float): learning rate
            threshold (float): loss function threshold
            loss_fn (Callable): layer level loss function
            bias (bool, optional): if bias. Defaults to True.
        """
        super(FFConv2d, self).__init__(in_channels, out_channels, kernel_size, padding=1)

        self.conv = Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias, padding=1)
        self.activation = activation
        self.optimizer = optimizer(self.parameters(), lr=layer_optim_learning_rate)
        self.threshold = threshold
        self.loss_fn = loss_fn
        self.method = method

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Model forward function

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: output tensor
        """
        x = x / (x.norm(2, 1, keepdim=True) + 1e-8)  # normalize input
        x = self.conv(x)
        return self.activation(x)

    def train_layer(self, X_pos: torch.Tensor, X_neg: torch.Tensor, before: bool) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """Train layer with FF algorithm

        Args:
            X_pos (torch.Tensor): batch of positive examples
            X_neg (torch.Tensor): batch of negative examples
            before (bool): if True, successive layers get previous layers output before update

        Returns:
            Tuple[torch.Tensor, torch.Tensor, int]: batch of positive and negative predictions and loss value
        """
        X_pos_out = self.forward(X_pos)
        X_neg_out = self.forward(X_neg)

        loss = self.loss_fn(X_pos_out, X_neg_out, self.threshold, self.method)

        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        if before:
            return X_pos_out.detach(), X_neg_out.detach(), loss.item()
        else:
            return self.forward(X_pos).detach(), self.forward(X_neg).detach(), loss.item()


class FFMaxPool2d(MaxPool2d):
    """Max pooling layer Forward-Forward compatible
    """

    def train_layer(self, X_pos: torch.Tensor, X_neg: torch.Tensor, before: bool) -> torch.Tensor:
        """Train layer (dummy implementation)

        Args:
            X_pos (torch.Tensor): batch of positive examples
            X_neg (torch.Tensor): batch of negative examples
            before (bool): if True, successive layers get previous layers output before update

        Returns:
            torch.Tensor: input tensor
        """
        if before:
            return X_pos.detach(), X_neg.detach(), 0
        else:
            return self.forward(X_pos).detach(), self.forward(X_neg).detach(), 0