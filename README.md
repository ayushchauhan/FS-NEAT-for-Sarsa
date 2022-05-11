## FS-NEAT-for-Sarsa

We use NEAT (NeuroEvolution of Augmenting Topologies)-based approach to perform feature selection in [OpenAI Gym MountainCar-v0](https://gym.openai.com/envs/MountainCar-v0/) environment. We use Tile Coding and Radial Basis Functions for featurizing the input state space. The selected features are then used to learn a policy with True Online Sarsa($$\lambda$$). The value function approximator used there is linear.

#### Installation
Create a conda environment with the [environment.yml](https://github.com/ayushchauhan/FS-NEAT-for-Sarsa/blob/main/environment.yml) file provided. Then install the [neat-python](https://neat-python.readthedocs.io/en/latest/index.html) package using pip or from source. You also need to install pytorch to run the reinforce algorithm.

#### Repository Description
The repository is inspired from the example implementations present in the public [neat-python](https://github.com/CodeReclaimers/neat-python) library. The feature_selection directory contains the code to run NEAT with a feedforward neural network model. It contains the following files:
* config-feedforward - The config file for the NEAT algorithm. Important parameters to play with are population size (pop_size), initial connection and num hidden nodes. For more details refer the neat-python documentation.
* evolve_feedforward.py - The main file that runs the NEAT algorithm. It creates a population of genomes, evaluates them by featurizing the input space and using the return as the fitness function, determines the winner network and writes the features selected by it to a file.
* test_feedforward.py - Loads the winner network and tests it on a new episode. Also renders the mountain car video.
* visualize.py - Some helper functions to view the final learnt network.

The other two directories contain the respective algorithms with a run file each that reads the selected features and runs the corresponding algorithm. The code for tile coding and radial basis coding is integrated in the sarsa.py and reinforce.py files. The default featurization used is RBF. In the run files, replace the line that reads the features by `features = 'all'` to run with all features.
