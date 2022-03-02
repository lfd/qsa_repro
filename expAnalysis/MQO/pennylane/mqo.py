#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
import pennylane as qml
from pennylane import qaoa
from collections import Counter
from qiskit_optimization import QuadraticProgram, translators
from docplex.mp.model import Model
import warnings
warnings.filterwarnings('ignore')


# In[2]:


## Logic for constructing the MQO model

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
    wl = calculate_wl(costs, epsilon)
    wm = calculate_wm(savings, wl)
    
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


# In[4]:


## Code based on https://pennylane.ai/qml/demos/tutorial_qaoa_intro.html

def qaoa_layer(gamma, alpha, cost_hamiltonian, mixer_hamiltonian):
    qaoa.cost_layer(gamma, cost_hamiltonian)
    qaoa.mixer_layer(alpha, mixer_hamiltonian)
    
def circuit(params, wires, depth, cost_hamiltonian, mixer_hamiltonian):
    for w in wires:
        qml.Hadamard(wires=w)
    qml.layer(qaoa_layer, depth, params[0], params[1], cost_hamiltonian=cost_hamiltonian, mixer_hamiltonian=mixer_hamiltonian)
    
def cost_function(params, wires, depth, cost_hamiltonian, mixer_hamiltonian):
    circuit(params, wires, depth, cost_hamiltonian, mixer_hamiltonian)
    return qml.expval(cost_hamiltonian)

def sample(gamma, alpha, wires, depth, cost_hamiltonian, mixer_hamiltonian, number_of_qubits):
    circuit([gamma, alpha], wires, depth, cost_hamiltonian, mixer_hamiltonian)
    return [qml.sample(qml.PauliZ(i)) for i in range(number_of_qubits)]


# In[5]:


def create_cost_hamiltonian(linear, quadratic, offset):
    coefficients = []
    observables = []
    if offset != 0:
        observables.append(qml.Identity(0))
        coefficients.append(offset)
    for i in range(len(linear)):
        for j in range(len(linear)):
            if i > j: ## Avoid double entries
                continue
            if i == j and i in linear.keys(): 
                observables.append(qml.PauliZ(i))
                coefficients.append(linear[i])
            elif (i, j) in quadratic.keys():
                observables.append(qml.PauliZ(i) @ qml.PauliZ(j))
                coefficients.append(quadratic[(i, j)])
    cost_hamiltonian = qml.Hamiltonian(coefficients, observables)
    return cost_hamiltonian

def create_mixer_hamiltonian(wires):
    mixer_hamiltonian = qaoa.x_mixer(wires)
    return mixer_hamiltonian

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

def get_initial_params(p=1):
    params = []
    for i in range(2):
        initial_params = np.empty(p)
        initial_params.fill(0.5)
        params.append(initial_params.tolist())
    return params

## Solves the MQO problem consisting of the given queries, costs and savings with various algorithms
def solve_MQO(queries, costs, savings, p=1):   
    
    ## Construct the Ising model
    ising, offset = get_ising_model(queries, costs, savings)
    coeffs = np.real(ising.primitive.coeffs)
    pauli_array = ising.primitive.settings['data'].array
    
    # Create coefficient lists by parsing the pauli array
    linear, quadratic = get_coefficients_from_Pauli_array(pauli_array, coeffs)
    
    ## Construct the QAOA circuit
    wires = range(len(costs))
    cost_hamiltonian = create_cost_hamiltonian(linear, quadratic, offset)
    mixer_hamiltonian = create_mixer_hamiltonian(wires)
    
    dev = qml.device("default.qubit", wires=wires)
    circuit = qml.QNode(cost_function, dev)
        
    params = get_initial_params(p)
        
    ## Search for the optimal parameters
        
    optimizer = qml.GradientDescentOptimizer()
    steps = 200
        
    for i in range(steps):
        if i % 50 == 0:
            print("Step: " + str(i))
        params = optimizer.step(circuit, params, wires=wires, depth=p, cost_hamiltonian=cost_hamiltonian, mixer_hamiltonian=mixer_hamiltonian)
        
    ## Sample the circuit parameterized with the optimal parameters
    
    dev = qml.device("default.qubit", wires=wires, shots=10000)
    sample_circuit = qml.QNode(sample, dev)
   
    response = sample_circuit(gamma=params[0], alpha=params[1], wires=wires, depth=p, cost_hamiltonian=cost_hamiltonian, mixer_hamiltonian=mixer_hamiltonian, number_of_qubits=len(linear))   


# In[6]:


queries, costs, savings = get_sample_MQO_instance()
solve_MQO(queries, costs, savings)

