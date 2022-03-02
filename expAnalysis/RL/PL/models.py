import pennylane as qml
from tensorflow import keras

import config

# Setup Keras to use 64-bit floats (required by PennyLane)
keras.backend.set_floatx('float64')

@qml.qnode(qml.device("default.qubit", wires=config.n_qubits), interface='tf', diff_method='backprop')
def circuit(inputs, weights):

    # input part
    for i in range(config.n_qubits):
        qml.RX(inputs[i], wires=i)

    for i in range(config.n_layers):
        generate_layer(weights[i], config.n_qubits)

    return [qml.expval(qml.PauliZ(0) @ qml.PauliZ(1)), qml.expval(qml.PauliZ(2) @ qml.PauliZ(3))]

def generate_layer(params, n_qubits): 
    # variational part
    for i in range(n_qubits):
        qml.RY(params[i][0], wires=i)
        qml.RZ(params[i][1], wires=i)

    # entangling part
    for i in range(n_qubits):
        qml.CZ(wires=[i, (i+1) % n_qubits])

VQC_Layer = qml.qnn.KerasLayer(
                qnode=circuit,
                weight_shapes={'weights': (config.n_layers, config.n_qubits, 2)},
                weight_specs = {"weights": {"initializer": "Zeros"}},
                output_dim=2,
                name='VQC_Layer')