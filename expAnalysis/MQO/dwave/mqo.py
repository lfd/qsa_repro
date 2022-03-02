#!/usr/bin/env python
# coding: utf-8

# In[1]:

import numpy as np
import itertools
import dimod
from dimod.reference.samplers import ExactSolver
from neal.sampler import SimulatedAnnealingSampler
import warnings
warnings.filterwarnings('ignore')


# In[3]:


def gen_sum(p1, savings):
    return sum([s for ((i,j),s) in savings if i==p1])

def calculate_wl(costs, epsilon):
    return max(costs)+epsilon

def calculate_wm(savings, wl):
    if not savings:
        return wl
    return wl + max([gen_sum(p1, savings) for p1 in list({i for ((i,j), s) in savings})])

def get_linear_terms(costs, wl):
    linear = {}
    for i in range(len(costs)):
        linear['p'+str(i)] = -wl + costs[i]
    return linear

def get_quadratic_terms(queries, costs, savings, wm):
    quadratic = {}
    
    ## Initialize dictionary of quadratic terms with 0 weights
    for (i,j) in itertools.combinations(np.arange(0, len(costs)), 2):
        quadratic[('p'+str(i), 'p'+str(j))] = 0
    
    ## Add penalty weight wm to each pair of alternative plans for the same query 
    for k in queries.keys():
        for (i,j) in itertools.combinations(queries[k], 2):
            quadratic[('p'+str(i), 'p'+str(j))] = quadratic[('p'+str(i), 'p'+str(j))] + wm
    
    ## Add negative savings value for each pair of plans with possible savings
    for ((i,j), s) in savings:
        quadratic[('p'+str(i), 'p'+str(j))] = quadratic[('p'+str(i), 'p'+str(j))] - s
    
    return quadratic

def initialize_model(queries, costs, savings):
    epsilon=0.25
    wl = calculate_wl(costs, epsilon)
    wm = calculate_wm(savings, wl)

    bqm = dimod.BinaryQuadraticModel(get_linear_terms(costs, wl), get_quadratic_terms(queries, costs, savings, wm), 
                                     0, dimod.Vartype.BINARY)
    return bqm


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


def solve_mqo(queries, costs, savings):
    
    ## Initialize the model
    
    model = initialize_model(queries, costs, savings)

    ## Solve via simulated annealing
    
    sim_annealing_sampler = SimulatedAnnealingSampler()
    sim_annealing_response = sim_annealing_sampler.sample(model, beta_range=[0.1, 4.2], num_reads=1, 
                                                      num_sweeps=1000, beta_schedule_type='geometric')
    print("Results:")
    print(sim_annealing_response)


# In[6]:


queries, costs, savings = get_sample_MQO_instance()
solve_mqo(queries, costs, savings)




