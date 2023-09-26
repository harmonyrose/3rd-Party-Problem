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
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity

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

def rational_vote(self, voter):
         max_similarity = -1
         for candidate in self.candidates:
             similarity = cosine_similarity([voter.opinions], [candidate.opinions])[0][0]
             if similarity > max_similarity:
                 max_similarity = similarity
                 closest_candidate = candidate
         return closest_candidate, max_similarity
    
class Society(Model):
    def __init__(self, N, p, num_candidates):
        super().__init__()
        self.N = N
        self.p = p
        self.num_candidates = num_candidates
        self.graph = gen_graph(N, p)
        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        self.candidates = []
        
        for i in range(self.N):
            newVoter = Voter(i, self, list(np.random.uniform(0,1,num_opinions)),[])
            self.schedule.add(newVoter)
            # self.datacollector= DataCollector(model_reporters={"clusters":clusters})
    
        for i in range(self.num_candidates):
            newCandidate = Candidate(i, self, list(np.random.uniform(0,1,num_opinions)))
            self.candidates.append(newCandidate)
            
    def step(self):
        self.schedule.step()
        
    def elect(self):
        vote_counts = {candidate: 0 for candidate in self.candidates}
        cosine_similarities = []
        candidate_buckets = {candidate: [] for candidate in self.candidates}
        bucket_count = 0
        for voter in self.schedule.agents:
            chosen_candidate = rational_vote(self, voter)[0]
            cosine_similarities.append(rational_vote(self, voter)[1])
            vote_counts[chosen_candidate] += 1
            if voter.bucket not in candidate_buckets[chosen_candidate]:
                candidate_buckets[chosen_candidate].append(voter.bucket)
                bucket_count += 1
            
        return vote_counts, cosine_similarities, candidate_buckets
          
        
class Candidate(Agent):
    def __init__(self, unique_id, model, opinions):
        super().__init__(unique_id, model)
        self.opinions = opinions
        
        
# represents a voter with an array of opinions.
# at each step of the simulation, voter x may influence voter y's
# opinion on an issue if they agree on a different issue.
     
class Voter(Agent):
    def __init__(self, unique_id, model, opinions, bucket):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.bucket = bucket
        
    def step(self):
        #print(self.opinions)
        # retreives x's neighbors and chooses one at random, y.
        nbrs = list(self.model.graph.neighbors(self.unique_id))
        neighbor = self.model.schedule.agents[np.random.choice(nbrs)]
        # calculates the absolute difference between x and y's opinions
        # on a randomly chosen issue, i1.
        i1 = np.random.choice(np.arange(num_opinions))
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
            self.bucket = bucket
            
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
                    self.bucket = bucket
                    break

        #if the agent is bucketless, put it into its very own bucket
        in_a_bucket = False
        for bucket in buckets:
            if self in bucket:
                in_a_bucket = True
        if not in_a_bucket:
            bucket = []
            bucket.append(self)
            self.bucket = bucket
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
num_steps = 350
cluster_threshold = 0.05
buckets = []

num_candidates = 10
election_steps = 50


# generates a model society
soc = Society(N, edge_probability, num_candidates)

# completes 150 iterations of the simulation, prints the number of buckets at
# each iteration
i = 0
e = 0
num_buckets = np.empty(num_steps + 1, dtype=int)
while i <= num_steps:
    if i % election_steps == 0:
        e += 1
        votes = soc.elect()[0]
        cosine_similarities = soc.elect()[1]
        candidate_buckets = soc.elect()[2]
        candidate_names = [candidate.opinions for candidate in votes.keys()]
        f_candidates = []
        for candidate in candidate_names:
            f_candidate = ["{:.1f}".format(opinion) for opinion in candidate]
            f_candidates.append(f_candidate)
        formatted_strs = []
        for candidate in f_candidates:
            formatted_str = ", ".join(candidate) 
            formatted_strs.append(formatted_str)
        vote_values = list(votes.values())
        plt.bar(formatted_strs, vote_values)
        plt.xlabel("Candidates")
        plt.ylabel("Vote Count")
        plt.title(f"Election {e} Outcome")
        plt.xticks(rotation=45, ha="right")
        plt.show()
        
        plt.hist(cosine_similarities, bins=50, edgecolor='black', range=(0,1), color='purple')  # Adjust the number of bins as needed
        plt.xlabel('Cosine Similarity')
        plt.ylabel('Frequency')
        plt.title('Histogram of Voter Distances to Closest Candidate')
        plt.show()
    
        bucket_counts = []
        for candidate in candidate_buckets:
            bucket_counts.append(len(candidate_buckets[candidate]))
        plt.bar(formatted_strs, bucket_counts, color="green")
        plt.xlabel("Candidates")
        plt.ylabel("Number of Unique Buckets")
        plt.title("Number of Different Buckets Voting for Each Candidate")
        plt.xticks(rotation=45, ha="right")
        plt.show()
        
    soc.step()
    print(len(buckets))
    num_buckets[i] = len(buckets)
    

        
    i += 1

plt.figure()
plt.plot(num_buckets)
plt.ylabel("Number of buckets")
plt.xlabel("Simulation iteration")
plt.show()
