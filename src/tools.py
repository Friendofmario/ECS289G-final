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
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def goodness_fun(x, method):
    if method == "MSE":
        return x.pow(2).mean(1)
    elif method == "MAE":
        return x.abs().mean(1)
    elif method == "cos_sim":
        return 1 - nn.functional.cosine_similarity(x, x.mean(0, keepdim=True), dim=1)
    elif method == "logcosh":
        return torch.log(torch.cosh(x)).mean(1)
   # Hubers with different deltas
    elif method == "huber1":
        delta = 1
        return nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none', beta=delta).sum(1)
    elif method == "huber2":
        delta = 2
        return nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none', beta=delta).sum(1)
    elif method == "huber0.5":
        delta = 0.5
        return nn.functional.smooth_l1_loss(x, torch.zeros_like(x), reduction='none', beta=delta).sum(1)
        # Broken
#    elif method == "cross_entropy":
#        return F.cross_entropy(x, torch.argmax(x, dim=1))

def base_loss(X_pos: torch.Tensor, X_neg: torch.Tensor, th: float, method: str) -> torch.Tensor:
    """Base loss described in the paper. Log(1+exp(x)) is added to help differentiation.

    Args:
        X_pos (torch.Tensor): batch of positive model predictions
        X_neg (torch.Tensor): batch of negative model predictions
        th (float): loss function threshold

    Returns:
        torch.Tensor: output loss
    """
    logits_pos = goodness_fun(X_pos, method)#X_pos.pow(2).mean(dim=1)
    logits_neg = goodness_fun(X_neg, method)#X_neg.pow(2).mean(dim=1)

    loss_pos = - logits_pos + th
    loss_neg = logits_neg - th

    loss_poss = torch.log(1 + torch.exp(loss_pos)).mean()
    loss_neg = torch.log(1 + torch.exp(loss_neg)).mean()

    loss = loss_poss + loss_neg

    return loss


def generate_positive_negative_samples_overlay(X: torch.Tensor, Y: torch.Tensor, num_classes: int, only_positive: bool, replace: bool = True) -> Tuple[torch.Tensor]:
    """Generate positive and negative samples using labels. It overlays labels in input. For neg it does
    the same but with shuffled labels.

    Args:
        X (torch.Tensor): batch of samples
        Y (torch.Tensor): batch of labels
        only_positive (bool): if True, it outputs only positive exmples with labels overlayed

    Returns:
        Tuple[torch.Tensor]: batch of positive (and negative samples)
    """
    X_pos = X.clone()

    if (replace):
        X_pos[:, :num_classes] *= 0.0
        X_pos[range(X.shape[0]), Y] = X_pos.max()  # one hot
    else:
        X_pos = F.pad(X_pos, (num_classes, 0), 'constant', 0)
        X_pos[range(X.shape[0]), Y] = X_pos.max()  # one hot

    if only_positive:
        return X_pos
    else:
        X_neg = X.clone()
        rnd = torch.randperm(X_neg.size(0))
        Y_neg = Y[rnd]

        for i in range(Y_neg):
            while Y_neg[i] == Y[i]:
                Y_neg[i] = random.randint(num_classes)

        if (replace):
            X_neg[:, :num_classes] *= 0.0
            X_neg[range(X_neg.shape[0]), Y_neg] = X_neg.max()  # one hot
        else:
            X_neg = F.pad(X_neg, (num_classes, 0), 'constant', 0)
            X_neg[range(X.shape[0]), Y_neg] = X_neg.max()  # one hot

        return X_pos, X_neg
