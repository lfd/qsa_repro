#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import itertools
from qiskit.compiler import transpile
from qiskit import(QuantumCircuit, execute, Aer, BasicAer)
from qiskit.visualization import plot_histogram
from qiskit_optimization import QuadraticProgram
from qiskit_optimization.algorithms import GroverOptimizer
from docplex.mp.model import Model
from qiskit.utils import algorithm_globals, QuantumInstance
from qiskit.algorithms import NumPyMinimumEigensolver
import qiskit.algorithms
from qiskit_optimization.algorithms import MinimumEigenOptimizer, RecursiveMinimumEigenOptimizer
from qiskit import IBMQ
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
    return wl + max([gen_sum(p1, savings) 
        for p1 in list({i for ((i,j), s) in savings})])

def construct_model(model, queries, costs, savings):
    v = model.binary_var_list(len(costs))
    epsilon = 0.25
    wl= calculate_wl(costs, epsilon)
    wm= calculate_wm(savings, wl)
    
    El = model.sum(-1*(wl-costs[i])*v[i] 
                for i in range(0, len(costs)))
    Em = model.sum(model.sum(wm*v[i]*v[j] 
                for (i,j) in itertools.combinations(queries[k], 2)) 
                for k in queries.keys())
    Es = model.sum(-s*v[i]*v[j] for ((i,j), s) in savings)
    return(El + Em + Es)


# In[3]:


def solve_with_QAOA(qubo):
    qaoa_meas = qiskit.algorithms.QAOA(quantum_instance=Aer.get_backend('qasm_simulator'), initial_point=[0., 0.])
    qaoa = MinimumEigenOptimizer(qaoa_meas)
    qaoa_result = qaoa.solve(qubo)
    return qaoa_result, qaoa_meas.get_optimal_circuit()


# In[4]:


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


## Solves the MQO problem consisting of the given queries, costs and savings with various algorithms
def solve_MQO(queries, costs, savings):    
    model = Model('docplex_model')

    model.minimize(construct_model(model, queries, costs, savings))

    qubo = QuadraticProgram()
    qubo.from_docplex(model)

    result_QAOA, QAOA_circuit = solve_with_QAOA(qubo)

    print("QAOA evaluation:")
    print(result_QAOA)


# In[6]:


queries, costs, savings = get_sample_MQO_instance()
solve_MQO(queries, costs, savings)

