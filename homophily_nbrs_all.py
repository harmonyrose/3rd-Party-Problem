# -*- coding: utf-8 -*-
"""
Created on Fri Apr  7 00:45:39 2023

@author: harmo
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Apr  6 15:35:41 2023

@author: harmony peura
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 17 23:19:10 2023
@author: harmony peura
"""

from mesa import Model, Agent
from mesa.time import RandomActivation
from mesa.datacollection import DataCollector
from numpy.random import choice
import numpy as np
import matplotlib.pyplot as plt
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

def get_bucket_avg(bucket):
        bucket_avg = []
        for i in range(0, num_opinions):
            sum = 0
            for agent in bucket:
                sum += agent.opinions[i]
            avg = sum/(len(bucket))
            bucket_avg.append(avg)
        return bucket_avg
    
    
class Society(Model):
    def __init__(self, N, p):
        super().__init__()
        self.N = N
        self.p = p
        self.graph = gen_graph(N, p)
        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        for i in range(self.N):
            newVoter = Voter(i, self, list(np.random.uniform(0,1,num_opinions)))
            self.schedule.add(newVoter)
            # self.datacollector= DataCollector(model_reporters={"clusters":clusters})
        
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

        # choose a random issue, i1
        i1 = np.random.choice(np.arange(num_opinions))
        
        # get all neighbors' opinions on i1
        nbr_ids = list(self.model.graph.neighbors(self.unique_id))
        nbrs = []
        for nbr_id in nbr_ids:
            nbrs.append(self.model.schedule.agents[nbr_id])
        nbr_ops = []
        for nbr in nbrs:
            nbr_ops.append(nbr.opinions)

        # calculate difference between self's opinion and neighbors' opinions
        # on all opinions, then divide each one by the sum of all opinions to get
        # probability weights
        
        for i in range(len(nbr_ops)):
            for j in range(len(nbr_ops[i])):
                nbr_ops[i][j] -= self.opinions[j] 
                nbr_ops[i][j] = 1-abs(nbr_ops[i][j])

        nbr_ops = np.array(nbr_ops)
        s_ops = np.array(self.opinions)
        nbr_dists = []
        for i in range(len(nbr_ops)):
            nbr_dists.append(np.linalg.norm(s_ops - nbr_ops[i]))
        
        
        nbr_weights = nbr_dists/np.sum(nbr_dists)
        
        # choose a neighbor to interact with; neighbors with similar opinions
        # on i1 have a higher chance of being chosen
        neighbor = np.random.choice(nbrs, p=nbr_weights)
        
        # calcuate the difference between self and neighbor's opinion
        diff = abs(self.opinions[i1] - neighbor.opinions[i1])

        # chooses another random issue, i2, that is different from i1
        i2 = np.random.choice(np.arange(num_opinions))
        while i2 == i1:
            i2 = np.random.choice(np.arange(num_opinions))
        # if x and y agree on i1, then x's opinion on i2 will move
        # closer to y's opinion on i2
        if diff <= openness:
            self.opinions[i2] = (self.opinions[i2] + neighbor.opinions[i2])/2
            opinion_moved = True
        # if x and y strongly disagree on i1, then x's opinion on i2 will move
        # further from y's opinion on i2
        elif diff >= pushaway:
            if self.opinions[i2] >= 0.5:
                move_to = 1
            else:
                move_to = 0
            self.opinions[i2] = (self.opinions[i2] + move_to)/2
            opinion_moved = True
        else:
            opinion_moved = False
            
        # if this is the first agent, make a new bucket for it
        if len(buckets) == 0:
            bucket = []
            bucket.append(self)
            buckets.append(bucket)
            
        # remove the agent from its current bucket if its opinion changed
        if opinion_moved:
            for bucket in buckets:
                if self in bucket:
                    bucket.remove(self)
                    # if the bucket is now empty, remove it
                    if len(bucket) == 0:
                        buckets.remove(bucket)
                        
            # place agent in appropriate bucket         
            for bucket in buckets:
                avg_opinions = get_bucket_avg(bucket)
                belongs = True
                for i in np.arange(num_opinions):
                    if abs(self.opinions[i] - avg_opinions[i]) > cluster_threshold:
                        belongs = False
                if belongs:
                    bucket.append(self)
                    break

        #if the agent is bucketless, put it into its very own bucket
        in_a_bucket = False
        for bucket in buckets:
            if self in bucket:
                in_a_bucket = True
        if not in_a_bucket:
            bucket = []
            bucket.append(self)
            buckets.append(bucket)
                

             
            
# hyperparameters

# if two agents are within this threshold on a chosen opinion, then
# agent 1 "trusts" agent 2           
openness = 0.1
# if two agents are outside this threshold on a chosen opinion, then
# agent 1 "distrusts" agent 2
pushaway = 0.7
num_opinions = 5
N = 50
edge_probability = 0.5
num_steps = 150
cluster_threshold = 0.05
buckets = []


# generates a model society
soc = Society(N, edge_probability)

# completes 150 iterations of the simulation, prints the number of buckets at
# each iteration
i = 0
num_buckets = np.empty(num_steps, dtype=int)
while i < num_steps:
    soc.step()
    print(len(buckets))
    num_buckets[i] = len(buckets)
    i += 1

plt.figure()
plt.plot(num_buckets)
plt.ylabel("Number of buckets")
plt.xlabel("Simulation iteration")
plt.show()
