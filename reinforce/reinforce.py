from typing import Iterable
import numpy as np
from tqdm import tqdm
import torch
import torch.nn.functional as F


class StateFeatureVectorWithRBF():
    def __init__(self,
                 features,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions: int,
                 num_bases:np.array,
                 sigma2:np.array):

        self.state_low = state_low
        self.state_high = state_high
        self.num_bases = num_bases
        self.sigma2 = sigma2
        self.centers = []
        for i in range(len(self.state_low)):
            self.centers.append(list(np.linspace(self.state_low[i]+(self.state_high[i]-self.state_low[i])/(2*self.num_bases[i]),self.state_high[i]-(self.state_high[i]-self.state_low[i])/(2*self.num_bases[i]),self.num_bases[i])))

        self.features = features

    def feature_vector_len(self) -> int:
        if self.features == 'all':
            return int(np.prod(self.num_bases))
        return int(len(self.features))

    def __call__(self, s, done) -> np.array:

        if done:
            return np.zeros(self.feature_vector_len())

        if self.features == 'all':
            for i, dim_val in enumerate(s):
                features = np.array([np.exp(-(dim_val-center)**2/(2*self.sigma2[i])) for center in self.centers[i]])
                if i == 0:
                    feat_matrix = features
                else:
                    feat_matrix = np.dot(feat_matrix.reshape((-1,1)), features.reshape((1,-1)))

            feat_matrix[feat_matrix < 0.0001] = 0
            return feat_matrix.transpose().flatten()

        else:
            feat_vec_ = []
            for feature in self.features:
                feat = 1
                center_indices = feature.split('_')[1:]
                for i, dim_val in enumerate(s):
                    feat *= np.exp(-(dim_val - self.centers[i][int(center_indices[i])])**2/(2*self.sigma2[i]))
                if feat < 0.0001:
                    feat_vec_.append(0)
                else:
                    feat_vec_.append(feat)
            return np.array(feat_vec_) 



class VNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(VNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)    # 1st hidden layer
        self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)    # 2nd hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)    # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for 1st hidden layer
        x = F.relu(self.hidden2(x))     # activation function for 2nd hidden layer
        x = self.predict(x)             # linear output
        return x


class PiNet(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(PiNet, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)    # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)    # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))              # activation function for hidden layer
        x = F.softmax(self.predict(x))             # softmax output
        return x




class PiApproximationWithNN():
    def __init__(self,
                 state_dims,
                 num_actions,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        action_dims: the number of possible actions
        alpha: learning rate
        """
        self.n_features = state_dims
        self.n_hidden1 = 32
        self.n_hidden2 = 32
        self.outputs = num_actions
        self.net = PiNet(self.n_features, self.n_hidden1, self.n_hidden2, self.outputs)
        self.optim = torch.optim.Adam(self.net.parameters(), lr = alpha, betas=(0.9, 0.999))


    def __call__(self,s) -> int:
        self.net.eval()
        probs = self.net.forward(torch.tensor(s[None]).float()).cpu().detach().numpy()[0]
        return np.random.choice(list(range(self.outputs)), p = probs)

    def update(self, s, a, gamma_t, delta):
        """
        s: state S_t
        a: action A_t
        gamma_t: gamma^t
        delta: G-v(S_t,w)
        """
        self.net.train()
        prediction = self.net.forward(torch.tensor(s[None]).float())
        loss = torch.mul(torch.tensor([-gamma_t * delta]).float(), torch.log(prediction[0,a]))
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()

        return None

class Baseline(object):
    """
    The dumbest baseline; a constant for every state
    """
    def __init__(self,b):
        self.b = b

    def __call__(self,s) -> float:
        return self.b

    def update(self,s,G):
        pass

class VApproximationWithNN(Baseline):
    def __init__(self,
                 state_dims,
                 alpha):
        """
        state_dims: the number of dimensions of state space
        alpha: learning rate
        """
        self.n_features = state_dims
        self.n_hidden = 16
        self.net = VNet(self.n_features, self.n_hidden, 1)
        self.optim = torch.optim.Adam(self.net.parameters(), lr = alpha, betas=(0.9, 0.999))

    def __call__(self,s) -> float:
        self.net.eval()
        return self.net.forward(torch.tensor(s[None]).float()).cpu().detach().numpy()[0,0]

    def update(self,s,G):
        self.net.train()
        prediction = self.net.forward(torch.tensor(s[None]).float())
        loss = torch.mul(torch.square(prediction), 0.5) - torch.mul(prediction, torch.tensor([[G]]).float())
        self.optim.zero_grad()
        loss.backward()
        self.optim.step()
        return None


def REINFORCE(
    env, #open-ai environment
    gamma:float,
    num_episodes:int,
    pi:PiApproximationWithNN,
    V:Baseline,
    X) -> Iterable[float]:
    """
    implement REINFORCE algorithm with and without baseline.

    input:
        env: target environment; openai gym
        gamma: discount factor
        num_episode: #episodes to iterate
        pi: policy
        V: baseline
    output:
        a list that includes the G_0 for every episodes.
    """
    G0 = []

    for ep in tqdm(range(num_episodes)):
        episode = []
        S0 = env.reset()
        A0 = pi(X(S0,False))
        while True:
            S1, R1, done, info = env.step(A0)
            episode.append((S0, A0, R1, S1))
            if done:
                break
            S0 = S1
            A0 = pi(X(S0,done))
        returns = [0]
        for trans in episode[::-1]:
            returns.insert(0, gamma * returns[0] + trans[2])
        G0.append(returns[0])
        for idx, trans in enumerate(episode):
            delta = returns[idx] - V(trans[0])
            V.update(trans[0], returns[idx])
            pi.update(X(trans[0],False), trans[1], gamma**idx, delta)
        # print(list(pi.net.parameters()))
    return G0





