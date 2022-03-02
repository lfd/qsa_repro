import gym
from wrappers import CartPoleEncoding

## Environment
env = CartPoleEncoding(gym.make('CartPole-v0'))
val_env = CartPoleEncoding(gym.make('CartPole-v0'))

n_qubits = 4
n_layers = 5
num_steps = 50000
replay_capacity = 50000
epsilon_duration = 20000
epsilon_start = 1.0
epsilon_end = 0.01
gamma = 0.99
lr_vqc = 1e-3
lr_cl = 0.1
update_every = 30
train_after = 1000
train_every = 1
batch_size = 32
validate_every = 100
num_val_trials = 1
ibmq = False