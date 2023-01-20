# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:19:10 2023

@author: harmony peura
"""

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
import numpy as np
import networkx as nx

# generates an erdos renyi graph with N nodes and p edge probability.
# if the graph is not connected, a new graph is generated until a 
# connected one is obtained. 

def gen_graph(N, p):
    graph = nx.erdos_renyi_graph(N,p)
    while not nx.is_connected(graph):
        graph = nx.erdos_renyi_graph(N,p)
    return graph 

# represents a model society using an erdos renyi graph. each node is a voter
# with an array of opinions. each edge determines a voters neighbors.

class Society(Model):
    def __init__(self, N, p):
        super().__init__()
        self.N = N
        self.p = p
        self.graph = gen_graph(N, p)
        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        for i in range(self.N):
            newVoter = Voter(i, self, np.random.uniform(0.0, 1.0, num_opinions))
            self.schedule.add(newVoter)
        #self.datacollector= DataCollector(model_reporters=)
        
    def step(self):
        self.schedule.step()
    
# represents a voter with an array of opinions.
# at each step of the simulation, voter x may influence voter y's
# opinion on an issue if they agree on a different issue.
     
class Voter(Agent):
    def __init__(self, unique_id, model, opinions):
        super().__init__(unique_id, model)
        self.opinions = opinions
        
    def step(self):
        x = self
        # retreives x's neighbors and chooses one at random, y.
        nbrs = list(x.model.g.neighbors(x.unique_id))
        y = x.model.schedule.agents[np.random.choice(nbrs)]
        # calculates the absolute difference between x and y's opinions
        # on a randomly chosen issue, i1.
        i1 = np.random.choice(0, num_opinions-1)
        diff = abs(x.opinions[i1] - y.opinions[i1])
        # chooses another random issue, i2, that is different from i1
        i2 = np.random.choice(0, num_opinions-1)
        while i2 == i1:
            i2 = np.random.choice(0, num_opinions-1)
        # if x and y agree on i1, then x's opinion on i2 will move
        # closer to y's opinion on i2
        if diff <= openness:
            x.opinions[i2] = (x.opinions[i2] + y.opinions[i2])/2
        


openness = 0.4
num_opinions = 5
N = 50
edge_probability = 0.5

# generates a model society
soc = Society(N, edge_probability)

# completes 150 iterations of the simulation
i = 0
while i < 150:
    soc.step()
    i += 1
