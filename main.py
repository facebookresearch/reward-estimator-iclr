# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import copy
import glob
import os
import time
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from arguments import get_args
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize
from envs import make_env
from model import CNNPolicy
from storage import RolloutStorage
from configurations import load_params

args = get_args()

num_updates = int(args.num_frames) // args.num_steps // args.num_processes

torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)


def main():
    print("#######")
    print(
        "WARNING: All rewards are clipped or normalized so you need to use a monitor (see envs.py) or visdom plot to get true rewards")
    print("#######")

    if args.run_index is not None:
        load_params(args)

    try:
        os.makedirs(args.log_dir)
    except OSError:
        files = glob.glob(os.path.join(args.log_dir, '*.monitor.csv'))
        for f in files:
            os.remove(f)

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.environ['OMP_NUM_THREADS'] = '1'

    envs = [make_env(args.env_name, args.seed, i, args.log_dir)
            for i in range(args.num_processes)]

    if args.num_processes > 1:
        envs = SubprocVecEnv(envs)
    else:
        envs = DummyVecEnv(envs)

    if len(envs.observation_space.shape) == 1:
        envs = VecNormalize(envs)

    obs_shape = envs.observation_space.shape
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

    num_heads = 1 if args.reward_predictor else len(args.gamma)
    assert len(envs.observation_space.shape) == 3
    actor_critic = CNNPolicy(obs_shape[0], envs.action_space, use_rp=args.reward_predictor, num_heads=num_heads)

    assert envs.action_space.__class__.__name__ == "Discrete"
    action_shape = 1

    if args.cuda:
        actor_critic.cuda()

    if not args.reward_predictor:
        model_params = actor_critic.parameters()
    else:
        lrs = [args.lr_rp, args.lr]
        model_params = [{'params': model_p, 'lr': p_lr} for model_p, p_lr in zip(actor_critic.param_groups, lrs)]

    optimizer = optim.RMSprop(model_params, args.lr, eps=args.eps, alpha=args.alpha)

    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, actor_critic.state_size,
                              gamma=args.gamma, use_rp=args.reward_predictor)
    current_obs = torch.zeros(args.num_processes, *obs_shape)

    def update_current_obs(obs):
        shape_dim0 = envs.observation_space.shape[0]
        obs = torch.from_numpy(obs).float()
        if args.num_stack > 1:
            current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
        current_obs[:, -shape_dim0:] = obs

    obs = envs.reset()
    update_current_obs(obs)

    rollouts.observations[0].copy_(current_obs)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_obs = current_obs.cuda()
        rollouts.cuda()

    start = time.time()
    for j in range(num_updates):
        for step in range(args.num_steps):
            # Sample actions
            value, action, action_log_prob, states = actor_critic.act(
                Variable(rollouts.observations[step], volatile=True),
                Variable(rollouts.states[step], volatile=True),
                Variable(rollouts.masks[step], volatile=True))
            cpu_actions = action.data.squeeze(1).cpu().numpy()

            obs, raw_reward, done, info = envs.step(cpu_actions)

            if args.reward_noise > 0.0:
                stds = np.ones(raw_reward.shape) * args.reward_noise
                noise = np.random.normal(loc=0.0, scale=stds)
                reward = raw_reward + noise
            else:
                reward = raw_reward

            raw_reward = torch.from_numpy(np.expand_dims(np.stack(raw_reward), 1)).float()

            episode_rewards += raw_reward

            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()

            if args.reward_predictor:
                p_hat = min(args.rp_burn_in, j) / args.rp_burn_in
                estimate_reward = (1 - p_hat) * reward + p_hat * value[:, 0].unsqueeze(-1).data.cpu()
                reward = torch.cat([reward, estimate_reward], dim=-1)
                value = value.data
            else:
                value = value.data

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_obs.dim() == 4:
                current_obs *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_obs *= masks

            update_current_obs(obs)

            rollouts.insert(step, current_obs, states.data, action.data, action_log_prob.data, value, reward, masks)

        next_value = actor_critic(Variable(rollouts.observations[-1], volatile=True),
                                  Variable(rollouts.states[-1], volatile=True),
                                  Variable(rollouts.masks[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value)

        states = Variable(rollouts.states[0].view(-1, actor_critic.state_size))
        masks = Variable(rollouts.masks[:-1].view(-1, 1))
        obs = Variable(rollouts.observations[:-1].view(-1, *obs_shape))
        actions = Variable(rollouts.actions.view(-1, action_shape))

        values, action_log_probs, dist_entropy, states = actor_critic.evaluate_actions(obs,
                                                                                       states,
                                                                                       masks,
                                                                                       actions)
        returns_as_variable = Variable(rollouts.returns[:-1])

        values = values.view(returns_as_variable.size())
        action_log_probs = action_log_probs.view(args.num_steps, args.num_processes, 1)

        advantages = returns_as_variable - values

        value_loss = advantages.pow(2).sum(-1).mean()
        action_loss = -(Variable(advantages[:, :, -1].unsqueeze(-1).data) * action_log_probs).mean()

        optimizer.zero_grad()
        (value_loss * args.value_loss_coef + action_loss - dist_entropy * args.entropy_coef).backward()

        nn.utils.clip_grad_norm(actor_critic.parameters(), args.max_grad_norm)

        optimizer.step()

        rollouts.after_update()

        if j % args.save_interval == 0 and args.save_dir != "":
            save_path = os.path.join(args.save_dir, 'a2c')
            try:
                os.makedirs(save_path)
            except OSError:
                pass

            # A really ugly way to save a model to CPU
            save_model = actor_critic
            if args.cuda:
                save_model = copy.deepcopy(actor_critic).cpu()

            save_model = [save_model,
                          hasattr(envs, 'ob_rms') and envs.ob_rms or None]

            torch.save(save_model, os.path.join(save_path, args.env_name + ".pt"))

        if j % args.log_interval == 0:
            end = time.time()
            total_num_steps = (j + 1) * args.num_processes * args.num_steps
            print(
                "Updates {}, num timesteps {}, FPS {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, "
                "entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
                format(j, total_num_steps,
                       int(total_num_steps / (end - start)),
                       final_rewards.mean(),
                       final_rewards.median(),
                       final_rewards.min(),
                       final_rewards.max(), dist_entropy.data[0],
                       value_loss.data[0], action_loss.data[0]
                       ))


if __name__ == "__main__":
    main()
