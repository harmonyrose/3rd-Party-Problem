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

import mesa
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
import sys
from sklearn.preprocessing import StandardScaler
from mesa.time import RandomActivation
from mesa.batchrunner import batch_run
from mesa.datacollection import DataCollector
from itertools import product

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
         min_distance = 10
         for candidate in self.candidates:
             distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(voter.opinions, candidate.opinions)))
             if distance < min_distance:
                 min_distance = distance
                 closest_candidate = candidate
         return closest_candidate
    
class Society(mesa.Model):
    def __init__(self, N, p, max_iter, num_candidates, cluster_threshold, party_centroids):
        super().__init__()
        self.N = N
        self.p = p
        self.num_candidates = num_candidates
        self.cluster_threshold = cluster_threshold
        self.graph = gen_graph(N, p)
        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        self.candidates = []
        self.step_num = 0
        self.max_iter = max_iter
        self.party_centroids = {}

    
        for i in range(self.num_candidates):
            newCandidate = Candidate(i, self, list(np.random.uniform(0,1,num_opinions)), i)
            self.party_centroids[i] = newCandidate.opinions
            self.candidates.append(newCandidate)
            
        for i in range(self.N):
            newVoter = Voter(i, self, list(np.random.uniform(0,1,num_opinions)),[], 0)
            min_distance = 100
            closest_party = -1
            for party in self.party_centroids:
                distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(newVoter.opinions, self.party_centroids[party])))
                if distance < min_distance:
                    min_distance = distance
                    closest_party = party
            newVoter.party = closest_party
            self.schedule.add(newVoter)
            # self.datacollector= DataCollector(model_reporters={"clusters":clusters})

        self.datacollector = DataCollector(
            agent_reporters={},
            model_reporters={"election_results": Society.elect})
            
    def step(self):
        self.step_num += 1
        if self.step_num % 50 == 0:
            self.datacollector.collect(self)
        self.schedule.step()
    
    def recompute_centroids(self):
        for party in self.party_centroids:
            new_centroid = [0] * num_opinions
            party_members = 0
            for voter in self.schedule.agents:
                if voter.party == party:
                    party_members += 1
                    for i in range(len(voter.opinions)):
                        new_centroid[i] += voter.opinions[i]
            for i in range(len(new_centroid)):
                new_centroid[i] /= party_members
                i +=1
            self.party_centroids[party] = new_centroid
        
    def elect(self):
        vote_counts = {candidate.unique_id: 0 for candidate in self.candidates}
        distances = []
        candidate_buckets = {candidate.unique_id:
                                        [] for candidate in self.candidates}
        bucket_count = 0
        for voter in self.schedule.agents:
            if voter.should_vote():
                chosen_candidate = rational_vote(self, voter)
                vote_counts[chosen_candidate.unique_id] += 1
                if voter.bucket not in candidate_buckets[chosen_candidate.unique_id]:
                    candidate_buckets[chosen_candidate.unique_id].append(voter.bucket)
                    bucket_count += 1
        for candidate in self.candidates:
            print(f"candidate {candidate.unique_id}: {vote_counts[candidate.unique_id]}")
        # drift after election
        self.recompute_centroids()
        new_opinions = self.drift()
        for candidate in self.candidates:
            candidate.opinions = list(new_opinions[candidate.unique_id])
            #print(candidate.opinions)
        return list(vote_counts.values()) #, distances, candidate_buckets
    
    # candidates drift towards the opinions that will get them the most votes
    def drift(self):
        optimal_opinions = {candidate.unique_id: [] for candidate in self.candidates}
        for candidate in self.candidates:
            # create an array of offsets within ±0.1
            
            offset_range = np.linspace(-0.2, 0.2, 9)
            
            # generate all possible combinations of offsets
            offset_combinations = product(offset_range, repeat=len(candidate.opinions))
            
            # initialize an empty list to store the neighboring arrays
            neighboring_arrays = []

            
            for offsets in offset_combinations:
                # calculate neighboring arrays based on party centroid
                neighbor_array = np.array(self.party_centroids[candidate.party]) + np.array(offsets)
                neighboring_arrays.append(neighbor_array)
                
            # store all neighboring arrays in a dictionary
            neighboring_tuples = [tuple(arr) for arr in neighboring_arrays]
            vote_counts = {t: 0 for t in neighboring_tuples}
            
            # store the candidate's original opinions
            original_opinions = candidate.opinions
            for t in neighboring_tuples:
                candidate.opinions = t
                for voter in self.schedule.agents:
                    voting_agent = voter.should_vote()
                    # if this is a non-voting agent, ensure it will still be non-voting
                    # with the candidate's new opinions
                    if voting_agent == False:
                        distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(voter.opinions, candidate.opinions)))
                        if distance < no_vote_threshold:
                            voting_agent = True
                    # all voting agents vote 
                    if voting_agent:
                        chosen_candidate = rational_vote(self, voter)
                        if chosen_candidate == candidate:
                            vote_counts[t] += 1
                            
            # if the candidate got 0 votes, pick a random opinion array, given it is between 0-1
            if max(vote_counts.values()) == 0:
                for key in vote_counts:
                    if all(0 <= value <= 1 for value in key):
                        max_key = key
                        max_key_list = list(max_key)
                        break
            # if the candidate got more than 0 votes, retrieve the key (opinions) corresponding
            # to the max vote count. if any of the opinions are outside the 0-1 range, clip
            # them back to 0 or 1
            else:    
                max_key = max(vote_counts, key=lambda k: vote_counts[k])
                max_key_list = list(max_key)

                for i in range(len(max_key_list)):
                    if max_key_list[i] < 0:
                        max_key_list[i] = 0.0
                    elif max_key_list[i] > 1:
                        max_key_list[i] = 1.0
            
            # store the optimal opinions and set the candidate's opinions back to
            # what they were originally
            clipped_max_key = tuple(max_key_list)
            print(f"candidate {candidate.unique_id}: {clipped_max_key}") #" {vote_counts[max_key]}")
            clipped_max_key = list(clipped_max_key)
            optimal_opinions[candidate.unique_id] = clipped_max_key
            candidate.opinions = original_opinions
        return optimal_opinions

    def compute_SVD(self):
        X = np.r_[[ a.opinions for a in self.schedule.agents ]]
        # Center & standardize column means.
        X = StandardScaler().fit_transform(X)
        U, S, Vh = np.linalg.svd(X, hermitian=False)
        if "oldU" not in dir(self):
            self.oldU = U[:,:2]
        one = np.linalg.norm(self.oldU - U[:,:2] @ np.array([[1,0],[0,1]]))
        two = np.linalg.norm(self.oldU - U[:,:2] @ np.array([[1,0],[0,-1]]))
        three = np.linalg.norm(self.oldU - U[:,:2] @ np.array([[-1,0],[0,1]]))
        four = np.linalg.norm(self.oldU - U[:,:2] @ np.array([[-1,0],[0,-1]]))
        if one < two and one < three and one < four:
            self.oldU = U[:,:2] @ np.array([[1,0],[0,1]])
        if two < one and two < three and two < four:
            self.oldU = U[:,:2] @ np.array([[1,0],[0,-1]])
        if three < one and three < two and three < four:
            self.oldU = U[:,:2] @ np.array([[-1,0],[0,1]])
        if four < one and four < two and four < three:
            self.oldU = U[:,:2] @ np.array([[-1,0],[0,-1]])
        return self.oldU
    
    def plot(self):
        svecs = self.compute_SVD()
        plt.clf()
        nx.draw_networkx(self.graph,
            {N:(svecs[N,0],svecs[N,1]) for N in range(self.N)},
            node_color=[ a.opinions for a in self.schedule.agents ])
        plt.xlim((-1.2,1.2))
        plt.ylim((-1.2,1.2))
        plt.title(f"Time {self.step_num} of {self.max_iter}")
        plt.savefig(f"output{self.step_num:03}.png")
        plt.close()

        
              
class Candidate(mesa.Agent):
    def __init__(self, unique_id, model, opinions, party):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.party = party
        
# represents a voter with an array of opinions.
# at each step of the simulation, voter x may influence voter y's
# opinion on an issue if they agree on a different issue.
     
class Voter(mesa.Agent):
    def __init__(self, unique_id, model, opinions, bucket, party):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.bucket = bucket
        self.party = party
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
            if self.opinions[i2] < neighbor.opinions[i2]:
                self.opinions[i2] -= ((self.opinions[i2]) / 2)
            else:
                self.opinions[i2] += ((1-self.opinions[i2])/ 2)

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
                    if abs(self.opinions[i] - avg_opinions[i]) > self.model.cluster_threshold:
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
            
    def should_vote(self):
            min_distance = 100
            for candidate in self.model.candidates:
                distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(self.opinions, candidate.opinions)))
                if distance < min_distance:
                    min_distance = distance
            if min_distance > no_vote_threshold:
                return False
            else:
                return True
    

             
            
# hyperparameters

# if two agents are within this threshold on a chosen opinion, then
# agent 1 "trusts" agent 2           
openness = 0.1
# if two agents are outside this threshold on a chosen opinion, then
# agent 1 "distrusts" agent 2
pushaway = 0.6
num_opinions = 3
N = 20
edge_probability = 0.5
max_iter = 400
cluster_threshold = 0.05
buckets = []
termination = 10
no_vote_threshold = pushaway
party_centroids = {}

num_candidates = 3
election_steps = 50


# generates a model society


# completes 150 iterations of the simulation, prints the number of buckets at
# each iteration

def is_non_trivial(bucket):
    if len(bucket) > 2:
        return True
    else:
        return False
    


if __name__ == "__main__":

    if len(sys.argv) <= 1:
        sys.exit("Usage: CI2elections.py numSims.")

    num_sims = int(sys.argv[1])

    params = {
        "N": N,
        "p": edge_probability,
        "cluster_threshold": cluster_threshold,
        "num_candidates": num_candidates,
        "max_iter": max_iter,  # only needed for plot caption
        "party_centroids": party_centroids
    }

    if num_sims == 1:
        # Single run.
        s = Society(params["N"], params["p"], params["cluster_threshold"],
            params["num_candidates"], params["max_iter"], params["party_centroids"])
        for i in range(max_iter):
            s.step()
        single_results = s.datacollector.get_model_vars_dataframe()

        # You now have single_results in your environment. For example, you
        # could do:
        # >>> single_results.iloc[0].election_results
        # to see the results at time=0.

    else:

        batch_results = batch_run(Society,
            parameters=params,
            iterations=num_sims,
            max_steps=max_iter,
            number_processes=None,   # make this 1 to use only one CPU
            data_collection_period=election_steps
            )

        batch_results = pd.DataFrame(batch_results)

        # You now have batch_results in your environment. For example, you
        # could do:
        # >>> batch_results.iloc[0].election_results
        # to see the results of the first simulation in the suite, at time=0.
        # (See other columns in batch_results to explain what each line
        # signifies.)


