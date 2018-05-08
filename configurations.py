# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

GAMES = ['BeamRiderNoFrameskip-v4',
         'SpaceInvadersNoFrameskip-v4',
         'BreakoutNoFrameskip-v4',
         'PongNoFrameskip-v4',
         'QbertNoFrameskip-v4',
         'SeaquestNoFrameskip-v4']

SEEDS = [62, 43, 97]

NOISE = [0.0, 0.1, 0.2, 0.3, 0.4]

# gamma, name, rp
RUNS = [
        ([0.0, 0.99], "RewardPredictor", True),
        ([0.99], "Baseline", False),
        ([0.0, 0.99], "Baseline+", False)
       ]

RUN_ID = []

for seed in SEEDS:
    for game in GAMES:
        for noise in NOISE:
            for (gamma, name, rp) in RUNS:
                RUN_ID.append((seed, game, gamma, name, rp, noise))


def load_params(args):
    args.seed, args.env_name, args.gamma, args.name, args.reward_predictor, args.reward_noise = RUN_ID[args.run_index]
    args.log_dir = args.log_dir + args.env_name + '_' + str(args.seed) + '_' + args.name + str(args.reward_noise)
    args.save_dir = args.log_dir
