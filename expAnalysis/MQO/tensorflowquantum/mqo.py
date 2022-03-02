#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import tensorflow as tf
import tensorflow_quantum as tfq
import cirq
import cirq_google
import sympy

from qiskit_optimization import QuadraticProgram, translators
from docplex.mp.model import Model
import warnings
warnings.filterwarnings('ignore')


# In[2]:


## Constructing the MQO model

def gen_sum(p1, savings):
    return sum([s for ((i,j),s) in savings if i==p1])

def calculate_wl(costs, epsilon):
    return max(costs)+epsilon

def calculate_wm(savings, wl):
    if not savings:
        return wl
    return wl + max([gen_sum(p1, savings) for p1 in list({i for ((i,j), s) in savings})])

## Creates the QUBO model for the given MQO problem instance and returns the corresponding Ising model
def get_ising_model(queries, costs, savings):
    
    model = Model('docplex_model')

    v = model.binary_var_list(len(costs))
    epsilon = 0.25
    wl= calculate_wl(costs, epsilon)
    wm= calculate_wm(savings, wl)
    
    El = model.sum(-1*(wl-costs[i])*v[i] for i in range(0, len(costs)))
    Em = model.sum(model.sum(wm*v[i]*v[j] for (i,j) in itertools.combinations(queries[k], 2)) 
                   for k in queries.keys())
    Es = model.sum(-s*v[i]*v[j] for ((i,j), s) in savings)
    
    model.minimize(El+Em+Es)
        
    qubo = translators.from_docplex_mp(model)
    
    return qubo.to_ising()


# In[3]:


# Returns the queries, costs and savings collections for a small sample mqo instance
def get_sample_MQO_instance():
    ## Format: query id=key, list of associated query plans=value
    queries={0: [0,1,2], 1: [3,4], 2: [5,6,7]}

    ## Costs for each query plan. Format: index=plan, value=cost
    costs=[12,16,14,11,11,16,15,19]

    ## Cost savings for re-usable plans. Format: list of ((p1, p2), saving)
    ## where p1,p2 are overlapping plans for different queries
    savings=[((0,3), 40), ((0,4), 31),
            ((0,6), 4), ((0,7), 6),
            ((1,6), 2), ((1,7), 5)]
    return queries, costs, savings


# In[5]:


## Parses the given pauli array and returns dictionaries associating qubits and qubit pairs with their corresponding Ising coefficients
def get_coefficients_from_Pauli_array(pauli_array, coefficients):
    n = pauli_array.shape[1]
    offset = int(0.5*n)
    linear = {}
    quadratic = {}
    for i in range(pauli_array.shape[0]):
        qubits = []
        for j in range(0, offset):
            pauli_z_index = j + offset
            if pauli_array[i][pauli_z_index] == True:
                qubits.append(j)    
        if len(qubits) == 1:
            linear[qubits[0]] = coefficients[i]
        else:
            quadratic[(qubits[0], qubits[1])] = coefficients[i]
    return linear, quadratic

def get_hadamard_circuit(cirq_qubits):
    hadamard_circuit = cirq.Circuit()
    for node in range(len(cirq_qubits)):
        qubit = cirq_qubits[node]
        hadamard_circuit.append(cirq.H.on(qubit))
    return hadamard_circuit

def create_cost_hamiltonian(cirq_qubits, linear, quadratic, offset):
    pauli_string = float(offset)
    for k in linear.keys():
        qubit = cirq_qubits[k]
        pauli_string += cirq.PauliString(float(linear[k])*cirq.Z(qubit))
    for (i, j) in quadratic.keys():
        qubit1 = cirq_qubits[i]
        qubit2 = cirq_qubits[j]
        pauli_string += cirq.PauliString(float(quadratic[(i, j)])*(cirq.Z(qubit1)*cirq.Z(qubit2))) 
    return pauli_string

def create_mixer_hamiltonian(cirq_qubits, linear):
    mixer_ham = 0
    for node in range(len(linear)):
        qubit = cirq_qubits[node]
        mixer_ham += cirq.PauliString(cirq.X(qubit))
    return mixer_ham


## Solves the MQO problem consisting of the given queries, costs and savings with various algorithms
def solve_MQO(queries, costs, savings, p=1):
    ## Construct the Ising model
    ising, offset = get_ising_model(queries, costs, savings)
    coeffs = np.real(ising.primitive.coeffs)
    pauli_array = ising.primitive.settings['data'].array
    
    # Create coefficient lists by parsing the pauli array
    linear, quadratic = get_coefficients_from_Pauli_array(pauli_array, coeffs)
    
    ## Construct the QAOA circuit
    cirq_qubits = cirq.GridQubit.rect(1, len(costs))
    num_param = 2 * p
    qaoa_parameters = np.array(sympy.symbols("q0:%d"%num_param))
    hamiltonians = []
    for i in range(p):
        hamiltonians.append(create_cost_hamiltonian(cirq_qubits, linear, quadratic, offset))
        hamiltonians.append(create_mixer_hamiltonian(cirq_qubits, linear))
    qaoa_circuit = tfq.util.exponential(operators=hamiltonians, coefficients=qaoa_parameters)
    
    hadamard_circuit = get_hadamard_circuit(cirq_qubits)
    ## Build the keras model
    model_circuit, model_readout = qaoa_circuit, create_cost_hamiltonian(cirq_qubits, linear, quadratic, offset)
    input_ = [hadamard_circuit]
    input_ = tfq.convert_to_tensor(input_)
    optimum = [0]
    
    optimum = np.array(optimum)
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(), dtype=tf.dtypes.string))
    model.add(tfq.layers.PQC(model_circuit, model_readout, backend=cirq.Simulator()))
    
    ## Compile and train the model
    model.compile(loss=tf.keras.losses.mean_absolute_error, optimizer=tf.keras.optimizers.Adam())
    history = model.fit(input_, optimum, epochs=1, verbose=0)
    
    ## Extract the trained parameters and sample the parameterized circuit
    trained_params = model.trainable_variables
    
    sample_circuit = tfq.layers.AddCircuit()(input_, append=qaoa_circuit)
    results = tfq.layers.Sample()(sample_circuit, symbol_names=qaoa_parameters.tolist(), symbol_values=trained_params, repetitions=10000)


# In[6]:


queries, costs, savings = get_sample_MQO_instance()
solve_MQO(queries, costs, savings, p=1)

