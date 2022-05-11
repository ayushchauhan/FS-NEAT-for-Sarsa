import numpy as np
import gym
from sarsa import SarsaLambda, StateActionFeatureVectorWithTile, StateActionFeatureVectorWithRBF
import time
import matplotlib.pyplot as plt
from tqdm import tqdm

def test_sarsa_lamda(features='all'):
    env = gym.make("MountainCar-v0")
    gamma = 1.

    # X = StateActionFeatureVectorWithTile(
    #     features,
    #     env.observation_space.low,
    #     env.observation_space.high,
    #     env.action_space.n,
    #     num_tilings=10,
    #     tile_width=np.array([.37,.029])
    # )
    X = StateActionFeatureVectorWithRBF(
        features,
        env.observation_space.low,
        env.observation_space.high,
        env.action_space.n,       
        np.array([20,20]),
        np.array([0.04, 0.00022])
        )


    st_time = time.time()
    weights, TD_errors, returns = SarsaLambda(env, gamma, 0.8, 0.01, X, 2000)
    print("Training Time = ", (time.time()-st_time)/60)
    plt.figure(0)
    plt.plot(range(2000), returns, linewidth=2)
    plt.xlabel('Episode', fontsize = 16)
    plt.ylabel('Return', fontsize = 16)
    plt.savefig("Training_Return.png", bbox_inches='tight')
    print("Average Training Return = ", np.mean(returns))

    plt.figure(1)
    plt.plot(range(2000), TD_errors, linewidth=2)
    plt.xlabel('Episode', fontsize = 16)
    plt.ylabel('Cummulative TD Error', fontsize = 16)
    plt.savefig("Training_Error.png", bbox_inches='tight')

    def greedy_policy(s,done, w):
        Q = [np.dot(w, X(s,done,a)) for a in range(env.action_space.n)]
        return np.argmax(Q)

    def _eval(w, render=False):
        s, done = env.reset(), False
        if render: env.render()

        G = 0.
        while not done:
            a = greedy_policy(s,done,w)
            s,r,done,_ = env.step(a)
            if render: env.render()

            G += r
        return G

    Avg_Test_returns = []
    Max_Test_returns = []
    for w in tqdm(weights):
        Gs = [_eval(w) for _ in  range(10)]
        Avg_Test_returns.append(np.mean(Gs))
        Max_Test_returns.append(np.max(Gs))
    # print("Average Test Return = ", np.mean(Gs))
    # print("Max Test Return = ", np.max(Gs))
    plt.figure(2)
    plt.plot(range(0,2000,10), Avg_Test_returns, label = 'Avg Return', linewidth=2)
    plt.plot(range(0,2000,10), Max_Test_returns, label = 'Max Return', linewidth=2)
    plt.xlabel('Training Episode', fontsize = 16)
    plt.ylabel('Test Return', fontsize = 16)
    plt.legend(loc = 'lower right', fontsize = 15)
    plt.savefig("Test_Return.png", bbox_inches='tight')

    _eval(weights[-1],True)


if __name__ == "__main__":
    features = list(np.load('../feature_selection/selected-features.npy'))
    # print(len(features))
    # features = 'all'
    test_sarsa_lamda(features)
