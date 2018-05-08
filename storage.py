# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import numpy as np
import torch


class RolloutStorage(object):
    def __init__(self, num_steps, num_processes, obs_shape, action_space, state_size, gamma, use_rp=False):
        self.observations = torch.zeros(num_steps + 1, num_processes, *obs_shape)
        self.states = torch.zeros(num_steps + 1, num_processes, state_size)
        if not use_rp:
            self.rewards = torch.zeros(num_steps, num_processes, 1)
        else:
            self.rewards = torch.zeros(num_steps, num_processes, 2)
        self.value_preds = torch.zeros(num_steps + 1, num_processes, len(gamma))
        self.returns = torch.zeros(num_steps + 1, num_processes, len(gamma))
        self.action_log_probs = torch.zeros(num_steps, num_processes, 1)
        if action_space.__class__.__name__ == 'Discrete':
            action_shape = 1
        else:
            action_shape = action_space.shape[0]
        self.actions = torch.zeros(num_steps, num_processes, action_shape)
        if action_space.__class__.__name__ == 'Discrete':
            self.actions = self.actions.long()
        self.masks = torch.ones(num_steps + 1, num_processes, 1)
        self.gamma = torch.from_numpy(np.array(gamma)).type(self.returns.type())
        self.gamma = self.gamma.view(1, -1).expand((num_processes, len(gamma)))

    def cuda(self):
        self.observations = self.observations.cuda()
        self.states = self.states.cuda()
        self.rewards = self.rewards.cuda()
        self.value_preds = self.value_preds.cuda()
        self.returns = self.returns.cuda()
        self.action_log_probs = self.action_log_probs.cuda()
        self.actions = self.actions.cuda()
        self.masks = self.masks.cuda()
        self.gamma = self.gamma.cuda()

    def insert(self, step, current_obs, state, action, action_log_prob, value_pred, reward, mask):
        self.observations[step + 1].copy_(current_obs)
        self.states[step + 1].copy_(state)
        self.actions[step].copy_(action)
        self.action_log_probs[step].copy_(action_log_prob)
        self.value_preds[step].copy_(value_pred)
        self.rewards[step].copy_(reward)
        self.masks[step + 1].copy_(mask)

    def after_update(self):
        self.observations[0].copy_(self.observations[-1])
        self.states[0].copy_(self.states[-1])
        self.masks[0].copy_(self.masks[-1])

    def compute_returns(self, next_value):
        self.returns[-1] = next_value
        for step in reversed(range(self.rewards.size(0))):

            self.returns[step] = self.returns[step + 1] * \
                                 self.gamma * self.masks[step + 1].expand_as(self.gamma) + \
                                 self.rewards[step].expand_as(self.gamma)