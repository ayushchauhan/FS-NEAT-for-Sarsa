import sys
import numpy as np
import gym
from matplotlib import pyplot as plt

from reinforce import REINFORCE, PiApproximationWithNN, Baseline, VApproximationWithNN, StateFeatureVectorWithRBF

def test_reinforce(features, with_baseline):
    env = gym.make("MountainCar-v0")
    gamma = 1.
    alpha = 3e-4

    X = StateFeatureVectorWithRBF(
        features,
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,
        np.array([15,15]),
        np.array([0.04, 0.00022])
        )

    pi = PiApproximationWithNN(
        X.feature_vector_len(),
        env.action_space.n,
        alpha)

    if with_baseline:
        B = VApproximationWithNN(
            env.observation_space.shape[0],
            alpha)
    else:
        B = Baseline(0.)

    return REINFORCE(env,gamma,1000,pi,B,X)

if __name__ == "__main__":
    num_iter = 5
    features = 'all'
    # features = list(np.load('../feature_selection/selected-features.npy'))
    
    # Test REINFORCE without baseline
    without_baseline = []
    for _ in range(num_iter):
        training_progress = test_reinforce(features, with_baseline=False)
        without_baseline.append(training_progress)
    without_baseline = np.mean(without_baseline,axis=0)

    # # Test REINFORCE with baseline
    # with_baseline = []
    # for _ in range(num_iter):
    #     training_progress = test_reinforce(with_baseline=True)
    #     with_baseline.append(training_progress)
    # with_baseline = np.mean(with_baseline,axis=0)

    # Plot the experiment result
    fig,ax = plt.subplots()
    ax.plot(np.arange(len(without_baseline)),without_baseline, label='without baseline')
    # ax.plot(np.arange(len(with_baseline)),with_baseline, label='with baseline')

    ax.set_xlabel('iteration')
    ax.set_ylabel('G_0')
    ax.legend()

    plt.show()
