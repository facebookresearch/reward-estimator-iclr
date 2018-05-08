# Reward Estimation for Variance Reduction in Deep RL

[Link to OpenReview submission](https://openreview.net/forum?id=r1vcHYJvM)

## Installation 

We based our code primarily off of [ikostrikov's pytorch-rl repo](https://github.com/ikostrikov/pytorch-a2c-ppo-acktr). Follow installation instructions there. 

## How to run

To replicate the exact results from the paper you need to run all 270 runs individually with:

``` python main.py --run-index [0-269] ```

To run the standard A2C (Baseline) on pong use the following:

``` python main.py --env-name PongNoFrameskip-v4 ```

To run A2C with the reward prediction auxilliary task (Baseline+) on pong use the following:

``` python main.py --env-name PongNoFrameskip-v4 --gamma 0.0 0.99 ```

To run A2C with reward prediction (Ours) on pong use the following:

``` python main.py --env-name PongNoFrameskip-v4 --reward-predictor --gamma 0.0 0.99 ```

## Visualization

run visualize.py to visualize performance (requires Visdom)

