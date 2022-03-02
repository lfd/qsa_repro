import sympy
from tensorflow import keras
import tensorflow as tf
import cirq
import tensorflow_quantum as tfq

class VQC_Layer(keras.layers.Layer):
    def __init__(self, n_qubits, n_layers):
        super(VQC_Layer, self).__init__()
        
        self.n_qubits = n_qubits
        self.qubits = cirq.GridQubit.rect(1, n_qubits)
        inputs = sympy.symbols(f'inputs0:{n_qubits}')
        params = sympy.symbols(f'param0:{n_qubits*n_layers*2}')
        self.params = tf.Variable(initial_value=tf.zeros((1,len(params))), dtype='float32', trainable=True, name='params')

        symbols = [str(symb) for symb in params + inputs]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])

        circuit = cirq.Circuit()

        # input part
        circuit.append([cirq.rx(inputs[i]).on(qubit) for i, qubit in enumerate(self.qubits)])

        for i in range(n_layers):
            circuit.append(self.generate_layer(params[i*n_qubits*2 : (i+1)*n_qubits*2]))

        readout_op = [cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[:2]),
                      cirq.PauliString(cirq.Z(qubit) for qubit in self.qubits[2:]) ]

        self.vqc = tfq.layers.ControlledPQC(circuit, readout_op, 
            differentiator=tfq.differentiators.ParameterShift())

    def generate_layer(self, params):
        circuit = cirq.Circuit()
        # variational part
        for i, qubit in enumerate(self.qubits):
            circuit.append([cirq.ry(params[i*2]).on(qubit),
                            cirq.rz(params[i*2+1]).on(qubit)])
        # entangling part
        for i in range(self.n_qubits):
            circuit.append(cirq.CZ.on(self.qubits[i], self.qubits[(i+1) % self.n_qubits]))

        return circuit

    def call(self, inputs):
        batch_dim = tf.gather(tf.shape(inputs), 0)
        tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
        tiled_up_thetas = tf.tile(self.params, multiples=[batch_dim, 1])
        joined_vars = tf.concat([tiled_up_thetas, inputs], axis=1)
        joined_vars = tf.gather(joined_vars, self.indices, axis=1)
        return self.vqc([tiled_up_circuits, joined_vars])