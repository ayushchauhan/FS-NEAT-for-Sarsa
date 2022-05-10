"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle
import numpy as np

import neat
import gym.wrappers
from evolve_feedforward import StateFeatureVectorWithTile, StateFeatureVectorWithRBF


# load the winner
with open('winner-feedforward', 'rb') as f:
    c = pickle.load(f)

print('Loaded genome:')
print(c)

# Load the config file, which is assumed to live in
# the same directory as this script.
local_dir = os.path.dirname(__file__)
config_path = os.path.join(local_dir, 'config-feedforward')
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     config_path)

net = neat.nn.FeedForwardNetwork.create(c, config)
env = gym.make('MountainCar-v0')

# X = StateFeatureVectorWithTile(
#             env.observation_space.low,
#             env.observation_space.high,
#             num_tilings=10,
#             tile_width=np.array([.451,.0351])
#         )

X = StateFeatureVectorWithRBF(
    env.observation_space.low,
    env.observation_space.high,
    np.array([15, 15]),
    np.array([0.04, 0.00022])
    )

s, done = env.reset(), False
env.render()
fitness = 0
while not done:
    # inputs = (s-env.observation_space.low)/(env.observation_space.high-env.observation_space.low)
    inputs = X(s, done)
    direction = net.activate(inputs)
    if direction [0] > 0.51:
        action = 2
    elif direction[0] < 0.49:
        action = 0
    else:
        action = 1
    next_s, reward, done, info = env.step(action)
    env.render()

    fitness += reward
    s = next_s


print('Car took {0:.1f} steps to cross the hill'.format(-fitness))
