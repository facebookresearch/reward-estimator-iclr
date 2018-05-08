# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import torch
import torch.nn as nn
import torch.nn.functional as F
from distributions import Categorical
from utils import orthogonal


if torch.cuda.is_available():
    dtype = torch.cuda.FloatTensor
    idtype = torch.cuda.LongTensor
else:
    dtype = torch.FloatTensor
    idtype = torch.LongTensor


def to_numpy(tensor):
    if torch.cuda.is_available():
        return tensor.data.cpu().numpy()
    else:
        return tensor.data.numpy()


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1 or classname.find('Linear') != -1:
        orthogonal(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0)


class FFPolicy(nn.Module):
    def __init__(self):
        super(FFPolicy, self).__init__()

    def forward(self, inputs, states, masks):
        raise NotImplementedError

    def act(self, inputs, states, masks, deterministic=False):
        value, x, states = self(inputs, states, masks)
        action = self.dist.sample(x, deterministic=deterministic)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, action)
        return value, action, action_log_probs, states

    def evaluate_actions(self, inputs, states, masks, actions):
        value, x, states = self(inputs, states, masks)
        action_log_probs, dist_entropy = self.dist.logprobs_and_entropy(x, actions)
        return value, action_log_probs, dist_entropy, states


class CNNPolicy(FFPolicy):
    def __init__(self, num_inputs, action_space, use_rp=False, num_heads=1):
        super(CNNPolicy, self).__init__()
        self.use_rp = use_rp
        self.conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 32, 3, stride=1)

        self.linear1 = nn.Linear(32 * 7 * 7, 512)

        self.critic_linear = nn.Linear(512, num_heads)

        num_outputs = action_space.n
        self.dist = Categorical(512, num_outputs)

        if use_rp:
            self.extra_conv1 = nn.Conv2d(num_inputs, 32, 8, stride=4)
            self.extra_conv2 = nn.Conv2d(32, 64, 4, stride=2)
            self.extra_conv3 = nn.Conv2d(64, 32, 3, stride=1)
            self.extra_hidden = nn.Linear(32 * 7 * 7, 512)
            self.extra_critics = nn.Linear(512, 1)
            len_params = len(list(self.extra_conv1.parameters()) + list(self.extra_conv2.parameters()) +
                             list(self.extra_conv3.parameters()) + list(self.extra_hidden.parameters()) +
                             list(self.extra_critics.parameters()))
            self.param_groups = [list(self.parameters())[-len_params:], list(self.parameters())[:-len_params]]
        else:
            self.param_groups = [list(self.parameters())]

        self.train()
        self.reset_parameters()


    @property
    def state_size(self):
        return 1

    def reset_parameters(self):
        self.apply(weights_init)

        relu_gain = nn.init.calculate_gain('relu')
        self.conv1.weight.data.mul_(relu_gain)
        self.conv2.weight.data.mul_(relu_gain)
        self.conv3.weight.data.mul_(relu_gain)
        self.linear1.weight.data.mul_(relu_gain)

        if hasattr(self, 'extra_conv1'):
            self.extra_conv1.weight.data.mul_(relu_gain)
            self.extra_conv2.weight.data.mul_(relu_gain)
            self.extra_conv3.weight.data.mul_(relu_gain)

        if hasattr(self, 'extra_hidden'):
            self.extra_hidden.weight.data.mul_(relu_gain)

        if self.dist.__class__.__name__ == "DiagGaussian":
            self.dist.fc_mean.weight.data.mul_(0.01)

    def forward(self, inputs, states, masks):
        x = self.conv1(inputs / 255.0)
        x = F.relu(x)

        x = self.conv2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = F.relu(x)

        conv_out = x.view(-1, 32 * 7 * 7)
        x = self.linear1(conv_out)
        x = F.relu(x)

        value = self.critic_linear(x)
        if self.use_rp:
            extra = self.extra_conv1(inputs / 255.0)
            extra = F.relu(extra)

            extra = self.extra_conv2(extra)
            extra = F.relu(extra)

            extra = self.extra_conv3(extra)
            extra = F.relu(extra)

            extra = extra.view(-1, 32 * 7 * 7)
            extra = self.extra_hidden(extra)
            extra = F.relu(extra)
            extra = self.extra_critics(extra)
            value = torch.cat([extra, value], dim=-1)

        return value, x, states
