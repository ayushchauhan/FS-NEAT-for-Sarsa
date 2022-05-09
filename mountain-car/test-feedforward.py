"""
Test the performance of the best genome produced by evolve-feedforward.py.
"""

import os
import pickle
import numpy as np

import neat
import gym.wrappers

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_tilings:int,
                 tile_width:np.array):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = [int(np.ceil((state_high[i]-state_low[i])/tile_width[i])+1) for i in range(len(state_low))]

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        num_tiles = 1
        for tiles in self.num_tiles:
            num_tiles *= tiles
        return self.num_tilings * num_tiles

    def __call__(self, s, done) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        feat_vec = np.zeros(tuple([self.num_tilings]+self.num_tiles))
        if done:
            return feat_vec.flatten()
        
        for tiling in range(self.num_tilings):
            feature = [tiling,]
            for i, dim_val in enumerate(s):
                start = self.state_low[i] - (tiling * self.tile_width[i]/self.num_tilings)
                feature.append(int((dim_val-start)/self.tile_width[i]))
            feat_vec[tuple(feature)] = 1

        return feat_vec.flatten()



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

X = StateActionFeatureVectorWithTile(
            env.observation_space.low,
            env.observation_space.high,
            num_tilings=10,
            tile_width=np.array([.451,.0351])
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
