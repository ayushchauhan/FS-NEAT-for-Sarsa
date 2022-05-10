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

class StateFeatureVectorWithTile():
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


class StateFeatureVectorWithRBF():
    def __init__(self,
                 state_low:np.array,
                 state_high:np.array,
                 num_bases:np.array,
                 sigma2:np.array):

        self.state_low = state_low
        self.state_high = state_high
        self.num_bases = num_bases
        self.sigma2 = sigma2
        self.centers = []
        for i in range(len(self.state_low)):
            self.centers.append(list(np.linspace(self.state_low[i]+(self.state_high[i]-self.state_low[i])/(2*self.num_bases[i]),self.state_high[i]-(self.state_high[i]-self.state_low[i])/(2*self.num_bases[i]),self.num_bases[i])))


        # for c2 in np.linspace(self.state_low[1]+(self.state_high[1]-self.state_low[1])/(2*self.num_bases[1]), self.state_high[1]-(self.state_high[1]-self.state_low[1])/(2*self.num_bases[1]), self.num_bases[1]):
        #     for c1 in np.linspace(self.state_low[0]+(self.state_high[0]-self.state_low[0])/(2*self.num_bases[0]), self.state_high[0]-(self.state_high[0]-self.state_low[0])/(2*self.num_bases[0]), self.num_bases[0]):
        #         self.centers.append(np.array([c1, c2]))


    def feature_vector_len(self) -> int:

        return np.prod(num_bases)

    def __call__(self, s, done) -> np.array:

        if done:
            return np.zeros(self.feature_vector_len())

        for i, dim_val in enumerate(s):
            features = np.array([np.exp(-(dim_val-center)**2/(2*self.sigma2[i])) for center in self.centers[i]])
            if i == 0:
                feat_matrix = features
            else:
                feat_matrix = np.dot(feat_matrix.reshape((-1,1)), features.reshape((1,-1)))
        # for center in self.centers:
        #     feat_vec.append(np.exp(-np.linalg.norm(np.array(s)-center)/(2*self.sigma)))
        feat_matrix[feat_matrix < 0.0001] = 0
        return feat_matrix.transpose().flatten()


# Use the NN network phenotype and the discrete actuator force function.
def eval_genome(genome, config):
    net = neat.nn.FeedForwardNetwork.create(genome, config)

    fitnesses = []

    for runs in range(runs_per_net):
        env = gym.make('MountainCar-v0')
        # X = StateFeatureVectorWithTile(
        #     env.observation_space.low,
        #     env.observation_space.high,
        #     num_tilings=10,
        #     tile_width=np.array([.451,.0351])
        # )
        X = StateFeatureVectorWithRBF(
            env.observation_space.low,
            env.observation_space.high,
            np.array([15, 15]),
            np.array([0.08, 0.00044])
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


def get_selected_features(path='winner-feedforward.gv'):

    with open(path) as file:
        contents = file.read()
        #print(type(contents))
        lines=contents.split('\n')
        # print(type(lines))
        tiles=[]
        for l in lines:
            if l.find("->")!=-1:
                if l.find('t') == 1:
                    tiles.append(l[1:6])
        # print(tiles)
        file.close()
    # print(len(tiles))
    tiles=list(set(tiles))
    return tiles


def run():
    env = gym.make('MountainCar-v0')
        # X = StateFeatureVectorWithTile(
        #     env.observation_space.low,
        #     env.observation_space.high,
        #     num_tilings=10,
        #     tile_width=np.array([.451,.0351])
        # )



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
    # X = StateFeatureVectorWithTile(
    #         env.observation_space.low,
    #         env.observation_space.high,
    #         num_tilings=10,
    #         tile_width=np.array([.451,.0351])
    #     )
        
    # for tiling in range(X.num_tilings):
    #     for i in range(X.num_tiles[0]):
    #         for j in range(X.num_tiles[1]):
    #             node_names[k] = "t"+str(tiling)+"_"+str(i)+str(j)
    #             k -= 1

    X = StateFeatureVectorWithRBF(
        env.observation_space.low,
        env.observation_space.high,
        np.array([15, 15]),
        np.array([0.08, 0.00044])
        )
    for i in range(X.num_bases[0]):
        for j in range(X.num_bases[1]):
            node_names[k] = "c_"+str(i)+str(j)
            k -= 1

    visualize.draw_net(config, winner, False, node_names=node_names)

    visualize.draw_net(config, winner, view=False, node_names=node_names,
                       filename="winner-feedforward.gv")
    # visualize.draw_net(config, winner, view=False, node_names=node_names,
    #                    filename="winner-feedforward-enabled-pruned.gv", prune_unused=True)
    selected_features = get_selected_features(path='winner-feedforward.gv')
    np.save('selected-features.npy',np.array(selected_features))

if __name__ == '__main__':
    run()
