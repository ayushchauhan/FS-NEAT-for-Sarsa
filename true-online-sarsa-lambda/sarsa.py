import numpy as np
from tqdm import tqdm

class StateActionFeatureVectorWithTile():
    def __init__(self,
                 features,
                 state_low:np.array,
                 state_high:np.array,
                 num_actions:int,
                 num_tilings:int,
                 tile_width:np.array
                 ):
        """
        state_low: possible minimum value for each dimension in state
        state_high: possible maimum value for each dimension in state
        num_actions: the number of possible actions
        num_tilings: # tilings
        tile_width: tile width for each dimension
        """
        self.state_low = state_low
        self.state_high = state_high
        self.num_actions = num_actions
        self.num_tilings = num_tilings
        self.tile_width = tile_width
        self.num_tiles = [int(np.ceil((state_high[i]-state_low[i])/tile_width[i])+1) for i in range(len(state_low))]
        num_tiles = 1
        for tiles in self.num_tiles:
            num_tiles *= tiles
        total_feats = self.num_actions * self.num_tilings * num_tiles

        if features == 'all':
            self.features_indices = list(range(total_feats))
        else:
            features_indices = []
            for feat in features:
                tiling = feat[1]
                i = feat[-2]
                j = feat[-1]
                features_indices.append(int(tiling)*25+int(i)*5+int(j))

            self.features_indices = features_indices
            if features_indices:
                features_indices = np.array(features_indices)
                for i in range(self.num_actions-1):
                    self.features_indices.extend(features_indices+(i+1)*total_feats/self.num_actions)

    def feature_vector_len(self) -> int:
        """
        return dimension of feature_vector: d = num_actions * num_tilings * num_tiles
        """
        return len(self.features_indices)


    def __call__(self, s, done, a) -> np.array:
        """
        implement function x: S+ x A -> [0,1]^d
        if done is True, then return 0^d
        """
        feat_vec = np.zeros(tuple([self.num_actions]+[self.num_tilings]+self.num_tiles))
        if done:
            return feat_vec.flatten()[self.features_indices]
        
        for tiling in range(self.num_tilings):
            feature = [a, tiling]
            for i, dim_val in enumerate(s):
                start = self.state_low[i] - (tiling * self.tile_width[i]/self.num_tilings)
                feature.append(int((dim_val-start)/self.tile_width[i]))
            feat_vec[tuple(feature)] = 1

        return feat_vec.flatten()[self.features_indices]

def SarsaLambda(
    env, # openai gym environment
    gamma:float, # discount factor
    lam:float, # decay rate
    alpha:float, # step size
    X:StateActionFeatureVectorWithTile,
    num_episode:int,
) -> np.array:
    """
    Implement True online Sarsa(\lambda)
    """

    def epsilon_greedy_policy(s,done,w,epsilon=.0):
        nA = env.action_space.n
        Q = [np.dot(w, X(s,done,a)) for a in range(nA)]

        if np.random.rand() < epsilon:
            return np.random.randint(nA)
        else:
            return np.argmax(Q)
    w = np.zeros((X.feature_vector_len()))
    epsilon = 0.1

    returns = []
    TD_errors = []
    weights = []
    for ep in tqdm(range(num_episode)):
        return_ = 0
        TD_error_ = 0
        curr_state = env.reset()
        curr_action = epsilon_greedy_policy(curr_state, False, w, epsilon)
        curr_feat = X(curr_state, False, curr_action)
        z = np.zeros((X.feature_vector_len()))
        action_val_old = 0
        while True:
            next_state, reward, done, info = env.step(curr_action)
            next_action = epsilon_greedy_policy(next_state, done, w, epsilon)
            next_feat = X(next_state, done, next_action)
            curr_action_val = np.dot(w, curr_feat)
            next_action_val = np.dot(w, next_feat)
            TD_error = reward + gamma * next_action_val - curr_action_val
            z = gamma * lam * z + (1 - alpha * gamma * lam * np.dot(z, curr_feat)) * curr_feat
            w = w + alpha * (TD_error + curr_action_val - action_val_old) * z - alpha * (curr_action_val - action_val_old) * curr_feat
            action_val_old = next_action_val
            curr_feat = next_feat
            curr_action = next_action
            return_ += reward
            TD_error_ += TD_error
            if done:
                break
        returns.append(return_)
        TD_errors.append(TD_error_)
        if ep%10 == 0:
            weights.append(w)
    return weights, TD_errors, returns
