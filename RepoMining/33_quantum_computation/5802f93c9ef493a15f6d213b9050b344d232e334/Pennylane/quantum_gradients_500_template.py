#! /usr/bin/python3
import sys
import pennylane as qml
from pennylane import numpy as np

# DO NOT MODIFY any of these parameters
a = 0.7
b = -0.3
dev = qml.device("default.qubit", wires=3)


def natural_gradient(params):
    """Calculate the natural gradient of the qnode() cost function.

    The code you write for this challenge should be completely contained within this function
    between the # QHACK # comment markers.

    You should evaluate the metric tensor and the gradient of the QNode, and then combine these
    together using the natural gradient definition. The natural gradient should be returned as a
    NumPy array.

    The metric tensor should be evaluated using the equation provided in the problem text. Hint:
    you will need to define a new QNode that returns the quantum state before measurement.

    Args:
        params (np.ndarray): Input parameters, of dimension 6

    Returns:
        np.ndarray: The natural gradient evaluated at the input parameters, of dimension 6
    """

    natural_grad = np.zeros(6)

    # QHACK #
    import warnings
    warnings.filterwarnings('ignore')

    fsm = np.zeros((6, 6))
    qnode(params)
    default = np.array(dev.state).conj().T
    weight_copy = np.copy(params)
    for i in range(6):
        for j in range(6):
            weight_copy[i] += np.pi/2
            weight_copy[j] += np.pi/2
            qnode(weight_copy)
            plus = dev.state
            weight_copy[j] -= np.pi
            qnode(weight_copy)
            minus_1 = dev.state
            weight_copy[i] -= np.pi
            weight_copy[j] += np.pi
            qnode(weight_copy)
            minus_2 = dev.state
            weight_copy[j] -= np.pi
            qnode(weight_copy)
            minus_3 = dev.state
            fsm[i][j] = 1/8 * (-np.dot(default, plus) * np.dot(default, plus).conj() + np.dot(default, minus_1) * np.dot(default, minus_1).conj() + \
                np.dot(default, minus_2) * np.dot(default, minus_2).conj() - np.dot(default, minus_3) * np.dot(default, minus_3).conj())
            weight_copy[i] = params[i]
            weight_copy[j] = params[j]
    
    weights = params
    s = np.pi/2
    gradient = np.zeros(6)
    weight_copy = np.copy(weights)
    for i in range(len(weights)):
        weight_copy[i] += s
        plus = qnode(weight_copy)
        weight_copy[i] -= (2 * s)
        minus = qnode(weight_copy)
        gradient[i] = (plus - minus)/(2 * np.sin(s)) 
        weight_copy[i] = weights[i]

    f_minus = np.linalg.inv(fsm)
    natural_grad = f_minus @ gradient

    # QHACK #

    return natural_grad


def non_parametrized_layer():
    """A layer of fixed quantum gates.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    qml.RX(a, wires=0)
    qml.RX(b, wires=1)
    qml.RX(a, wires=1)
    qml.CNOT(wires=[0, 1])
    qml.CNOT(wires=[1, 2])
    qml.RZ(a, wires=0)
    qml.Hadamard(wires=1)
    qml.CNOT(wires=[0, 1])
    qml.RZ(b, wires=1)
    qml.Hadamard(wires=0)


def variational_circuit(params):
    """A layered variational circuit composed of two parametrized layers of single qubit rotations
    interleaved with non-parameterized layers of fixed quantum gates specified by
    ``non_parametrized_layer``.

    The first parametrized layer uses the first three parameters of ``params``, while the second
    layer uses the final three parameters.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    non_parametrized_layer()
    qml.RX(params[0], wires=0)
    qml.RY(params[1], wires=1)
    qml.RZ(params[2], wires=2)
    non_parametrized_layer()
    qml.RX(params[3], wires=0)
    qml.RY(params[4], wires=1)
    qml.RZ(params[5], wires=2)


@qml.qnode(dev)
def qnode(params):
    """A PennyLane QNode that pairs the variational_circuit with an expectation value
    measurement.

    # DO NOT MODIFY anything in this function! It is used to judge your solution.
    """
    variational_circuit(params)
    return qml.expval(qml.PauliX(1))


if __name__ == "__main__":
    # DO NOT MODIFY anything in this code block

    # Load and process inputs
    params = sys.stdin.read()
    params = params.split(",")
    params = np.array(params, float)

    updated_params = natural_gradient(params)

    print(*updated_params, sep=",")