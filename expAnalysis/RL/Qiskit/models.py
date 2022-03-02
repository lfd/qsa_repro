import numpy as np
from qiskit_machine_learning.connectors import TorchConnector
from qiskit.circuit import ParameterVector, QuantumCircuit
from qiskit.utils import QuantumInstance
from qiskit import Aer, IBMQ
from qiskit_machine_learning.neural_networks import OpflowQNN, CircuitQNN
import torch
from torch.nn import Module
from qiskit.opflow.gradients import Gradient
from qiskit.opflow import StateFn, PauliSumOp, ListOp, AerPauliExpectation

import config

class VQC_Layer(Module):
    def __init__(self, n_qubits, n_layers, shots=1024, device = torch.device("cuda")):
        super(VQC_Layer, self).__init__()
        self.n_qubits = n_qubits
        self.shots = shots
        weight_params = ParameterVector('param', n_qubits*n_layers*2)
        input_params = ParameterVector('input', n_qubits)

        if config.ibmq:
            ## IBMQ.save_account("API TOKEN")   # can be done beforehand in e.g. Python console 
            provider =  IBMQ.load_account()
            q_backend = provider.get_backend('ibmq_belem') 
        else:
            q_backend = Aer.get_backend('qasm_simulator')

            if device == torch.device("cuda"):
                q_backend.set_options(device='GPU')

        qi = QuantumInstance(q_backend, shots=shots)

        self.circuit = QuantumCircuit(self.n_qubits)

        for i, input in enumerate(input_params):
            self.circuit.rx(input, i)

        for i in range(n_layers):
            self.generate_layer(weight_params[i*n_qubits*2 : (i+1)*n_qubits*2])

        if config.ibmq:
            qnn = CircuitQNN(circuit=self.circuit,
                    input_params=input_params,
                    weight_params=weight_params,
                    quantum_instance=qi,
                    gradient=Gradient(grad_method='param_shift'))

            self.decode_layer = Decode_Layer(n_qubits, shots, device)
        else:
            readout_op = ListOp([
                ~StateFn(PauliSumOp.from_list([('ZZII', 1.0)])) @ StateFn(self.circuit),
                ~StateFn(PauliSumOp.from_list([('IIZZ', 1.0)])) @ StateFn(self.circuit)])

            qnn = OpflowQNN(readout_op,
                        input_params=input_params,
                        weight_params=weight_params,
                        exp_val=AerPauliExpectation(),
                        quantum_instance=qi,
                        gradient=Gradient(grad_method='param_shift'))
        
        self.qnn = TorchConnector(qnn, initial_weights=torch.Tensor(np.zeros(n_qubits*n_layers*2)))

    def generate_layer(self, params):
        # variational part
        for i in range(self.n_qubits):
            self.circuit.ry(params[i*2], i)
            self.circuit.rz(params[i*2+1], i)

        # entangling part
        for i in range(self.n_qubits):
            self.circuit.cz(i, (i+1) % self.n_qubits)

    def forward(self, inputs):
        x = self.qnn(inputs)
        if config.ibmq:
            x = self.decode_layer(x)
        return x

class Decode_Layer(Module):

    def __init__(self, n_qubits, shots = 1024, device=torch.device("cuda")):
        super(Decode_Layer, self).__init__()
        self.device=device
        self.n_qubits=n_qubits
        self.shots=shots
        basis_states = torch.tensor([val for val in range(2**self.n_qubits)], device=device, requires_grad=False)
        mask = 2 ** torch.arange(self.n_qubits - 1, -1, -1).to(device, basis_states.dtype)
        self.basis_states = basis_states.unsqueeze(-1).bitwise_and(mask).ne(0).float()
        
        self.high_index = [list(),list()]
        self.low_index = [list(), list()]
        for i in range(0,int(self.n_qubits/2)):
            ind = i*2
            for j, state in enumerate(self.basis_states):
                if state[ind] == 1 and state[ind + 1]==1 or state[ind] == 0 and state[ind + 1]==0:
                    self.high_index[i].append(j)
            self.low_index[i] = np.arange(self.n_qubits**2)
            self.low_index[i] = np.array(list(set(self.low_index[i])-set(self.high_index[i])))

        self.high_index = torch.as_tensor(self.high_index)
        self.low_index = torch.as_tensor(self.low_index)

    def forward(self, input):
        batch_size = input.shape[0]
        expectation_values = torch.zeros(int(self.n_qubits/2), batch_size)    
        for i in range(0,int(self.n_qubits/2)):
            high_x = torch.gather(input, index=torch.tile(self.high_index[i], (batch_size, 1)), dim=1)
            low_x = torch.gather(input, index=torch.tile(self.low_index[i], (batch_size, 1)), dim=1)
            expectation_values[i] = torch.sum(high_x, dim=1) - torch.sum(low_x, dim=1)
        return torch.transpose(expectation_values, 0, 1)
        