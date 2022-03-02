from PT.algorithm import Algorithm
import config

import torch
import copy
from torch.nn import Module

from PT.scales import Scale
from Qiskit.models import VQC_Layer

device = torch.device("cpu")
class Model(Module):

    def __init__(self) -> None:
        super().__init__()
        self.qnn = VQC_Layer(config.n_qubits, config.n_layers, device=device)
        self.scale = Scale()
        
    def forward(self, inputs):
        x = self.qnn(inputs)
        return self.scale(x)

def create_models():
    policy_model = Model()
    policy_model.to(device)
    target_model = copy.deepcopy(policy_model)
    target_model.to(device)

    return policy_model, target_model

if __name__=='__main__':

    policy_model, target_model = create_models()

    optimizer_vqc = torch.optim.Adam(policy_model.qnn.parameters(), lr=config.lr_vqc)
    optimizer_cl = torch.optim.Adam(policy_model.scale.parameters(), lr=config.lr_cl)

    algorithm = Algorithm(  config.env,
                            config.val_env,
                            policy_model, 
                            target_model,
                            config.replay_capacity,
                            config.epsilon_duration,
                            config.epsilon_start,
                            config.epsilon_end,
                            config.gamma,
                            optimizer_vqc,
                            optimizer_cl,
                            torch.nn.MSELoss(),
                            config.num_steps,
                            config.update_every,
                            config.train_after,
                            config.train_every,
                            config.batch_size,
                            config.validate_every,
                            config.num_val_trials,
                            'Qiskit',
                            device)

    algorithm.train()
    