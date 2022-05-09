"""
Mountain Car experiment using a feed-forward neural network.
"""

import multiprocessing
import os
import pickle

import neat
import visualize
import numpy as np

import gym.wrappers

runs_per_net = 5

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


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make('MountainCar-v0')
        X = StateActionFeatureVectorWithTile(
            env.observation_space.low,
            env.observation_space.high,
            num_tilings=10,
            tile_width=np.array([.451,.0351])
        )
        fitness = 0.0
        done = False
        curr_state = env.reset()
        while not done:
            # inputs = (curr_state - env.observation_space.low) / (env.observation_space.high - env.observation_space.low)
            inputs = X(curr_state, done)
            direction = net.activate(inputs)
            if direction [0] > 0.51:
                action = 2
            elif direction[0] < 0.49:
                action = 0
            else:
                action = 1
            next_state, reward, done, info = env.step(action)

            fitness += reward
            curr_state = next_state

        fitnesses.append(fitness)

    return min(fitnesses)


def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        genome.fitness = eval_genome(genome, config)


def run():
    # Load the config file, which is assumed to live in
    # the same directory as this script.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config-feedforward')
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_path)

    pop = neat.Population(config)
    stats = neat.StatisticsReporter()
    pop.add_reporter(stats)
    pop.add_reporter(neat.StdOutReporter(True))

    pe = neat.ParallelEvaluator(multiprocessing.cpu_count(), eval_genome)
    winner = pop.run(pe.evaluate)

    # Save the winner.
    with open('winner-feedforward', 'wb') as f:
        pickle.dump(winner, f)

    print(winner)

    visualize.plot_stats(stats, ylog=True, view=False, filename="feedforward-fitness.svg")
    visualize.plot_species(stats, view=False, filename="feedforward-speciation.svg")

    # node_names = {-1: 'x', -2: 'v', 0: 'control'}
    node_names = {0:'control'}
    k = -1
    env = gym.make('MountainCar-v0')
    X = StateActionFeatureVectorWithTile(
            env.observation_space.low,
            env.observation_space.high,
            num_tilings=10,
            tile_width=np.array([.451,.0351])
        )
        
    for tiling in range(X.num_tilings):
        for i in range(X.num_tiles[0]):
            for j in range(X.num_tiles[1]):
                node_names[k] = "t"+str(tiling)+"_"+str(i)+str(j)
                k -= 1


    visualize.draw_net(config, winner, False, node_names=node_names)

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)


if __name__ == '__main__':
    run()
