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
import random
import os
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


# Returns the candidate number closest in opinion space to the agent passed.
# (This is called voting "rationally" since it is based purely on similarity of
# opinions, not party affiliation or any other distracting factor.)
def rational_vote(self, voter):
     min_distance = 10
     for candidate in self.candidates:
         distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(voter.opinions, candidate.opinions)))
         if distance < min_distance:
             min_distance = distance
             closest_candidate = candidate
     return closest_candidate

# Returns the candidate number that belongs to the same party as the voter
def party_vote(self, voter):
    for candidate in self.candidates:
        if voter.party == candidate.party:
            return candidate

# Function used at initialization to determine all voters' voting algorithms
# based on frac_rational. Sets self.voting_algorithm to either "rational"
# or "party" for every voter.
def determine_voting_algorithms(self):
    num_rational = int(frac_rational * N)
    rational_voters = random.sample(self.schedule.agents, num_rational)
    for voter in self.schedule.agents:
        if voter in rational_voters:
            voter.voting_algorithm = rational_vote
        else:
            voter.voting_algorithm = party_vote



# An agent-based model of a society with N voters and num_candidates political
# candidates for office. Other constructor parameters:
# p - probability of social connection between any two voters
# max_iter - the longest time the sim will run before being terminated
# party_switch_threshold - threshold for how close voters' opinions need to be
#    to a different party's centroid in order for them to switch to that party
# no_vote_threshold -- if an agent is no closer than this value to any
#   candidate in opinion space, it will sit out the election
# frac_rational -- the proportion of voters who will vote rationally (as
#   opposed to by party)
# election_steps -- hold an election every this number of steps
class Society(mesa.Model):
    def __init__(self, N, p, party_switch_threshold, num_candidates, max_iter,
        no_vote_threshold, frac_rational, frac_party, frac_ff1, frac_ff2, 
        chase_radius, election_steps, do_anim=False):

        super().__init__()
        self.N = N
        self.p = p
        self.num_candidates = num_candidates
        self.party_switch_threshold = party_switch_threshold
        self.no_vote_threshold = no_vote_threshold
        self.graph = gen_graph(N, p)
        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        self.candidates = []
        self.voters = []
        self.step_num = 0
        self.max_iter = max_iter
        self.party_centroids = {}
        self.frac_rational = frac_rational
        self.frac_party = frac_party
        self.frac_ff1 = frac_ff1
        self.frac_ff2 = frac_ff2
        self.chase_radius = chase_radius
        self.election_steps = election_steps
        self.do_anim = do_anim

        # Create new random candidates, one for each party, and initialize each
        # party's "centroid" to be not actually its centroid of voters, but its
        # candidate's opinion vector. (TODO Issue #1)
        for i in range(self.num_candidates):
            newCandidate = Candidate(i, self, list(np.random.uniform(0,1,num_opinions)), i)
            self.party_centroids[i] = newCandidate.opinions
            self.candidates.append(newCandidate)

        # Create random voters, and assign each one to the party whose centroid
        # (candidate; see above) it is closest to in Euclidean opinion space.
        for i in range(self.N):
            newVoter = Voter(i, self, list(np.random.uniform(0,1,num_opinions)), 0, 0)
            min_distance = 100
            closest_party = -1
            for party in self.party_centroids:
                distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(newVoter.opinions, self.party_centroids[party])))
                if distance < min_distance:
                    min_distance = distance
                    closest_party = party
            newVoter.party = closest_party
            self.voters.append(newVoter)
            self.schedule.add(newVoter)
        determine_voting_algorithms(self)
        self.datacollector = DataCollector(
            agent_reporters={},
            model_reporters={"rational_results": Society.rational_elect,
                             "election_results": Society.elect},
            tables={
                "party_switches": ["agent_id","old_party","new_party","iter"],
                "zero_votes": ["party","iter"]
            }
        )

    # Make each agent run, and hold an election if it's time to.
    def step(self):
        self.step_num += 1
        # TODO Issue #12
        #if self.step_num % self.election_steps == 0:
        self.datacollector.collect(self)
        self.schedule.step()
        if self.do_anim:
            self.plot()

    # Based on the current opinion vectors and party assignments of all agents,
    # (re-)compute the centroid opinion for each party.
    def recompute_centroids(self):
        for party in self.party_centroids:
            new_centroid = [0] * num_opinions
            party_members = 0
            for voter in self.schedule.agents:
                if voter.party == party:
                    party_members += 1
                    for i in range(len(voter.opinions)):
                        new_centroid[i] += voter.opinions[i]
            if party_members != 0:
                for i in range(len(new_centroid)):
                    new_centroid[i] /= party_members
                self.party_centroids[party] = new_centroid
            else:
                for candidate in self.candidates:
                    if candidate.party == party:
                        self.party_centroids[party] = candidate.opinions


    # Run an election. This involves having each agent vote according to their
    # own algorithm (e.g., party-based, rational) and also vote rationally
    # (regardless of algorithm) so we have election results on each. Then,
    # after votes are tabulated, have each candidate strategically chase()
    # towards a new opinion vector that will get them the most votes.
    #
    # Returns two lists of vote counts, order by candidate number: the first
    # contains "real" election results and the second contains the results
    # that would have been achieved had every voter voted rationally.
    def elect(self):
        real_vote_counts = {candidate.unique_id: 0 for candidate in self.candidates}
        if self.step_num % self.election_steps != 0:
            return list(real_vote_counts.values())
        # Have all agents vote based on their voting algorithm and store the
        # vote counts in real_vote_counts
        for voter in self.schedule.agents:
            chosen_candidate = voter.voting_algorithm(self, voter)
            real_vote_counts[chosen_candidate.unique_id] += 1

        # chase after election
        new_opinions = self.chase()
        for candidate in self.candidates:
            candidate.opinions = list(new_opinions[candidate.unique_id])
        return list(real_vote_counts.values())

    def rational_elect(self):
        rational_vote_counts = {candidate.unique_id: 0 for candidate in self.candidates}
        if self.step_num % self.election_steps != 0:
            return list(rational_vote_counts.values())
        # Have all agents vote rationally and store the vote counts in rational_vote_counts
        for voter in self.schedule.agents:
            chosen_candidate = rational_vote(self, voter)
            rational_vote_counts[chosen_candidate.unique_id] += 1
        return list(rational_vote_counts.values())

    # candidates chase towards the opinions that will get them the most votes
    def chase(self):
        optimal_opinions = {candidate.unique_id: [] for candidate in self.candidates}
        for candidate in self.candidates:
            # creates an array that essentially acts as the diameter of the
            # chase_space where the points are spaced out by 0.05
            num_points = int(self.chase_radius * 40) + 1
            chase_diameter = np.linspace((-1 * self.chase_radius), self.chase_radius, num_points)

            # Generate the Cartesian product of num_opinions copies of the
            # chase_diameter. This gives us an iterator of tuples, each of size
            # num_opinions, that represents a possible distance (in opinion
            # space) that this candidate will consider moving to in order to
            # get more votes.
            chase_points = product(chase_diameter, repeat=len(candidate.opinions))
            # Initialize an empty list to store all the opinion vectors this
            # candidate will consider.
            chase_space = []
            for chase_point in chase_points:
                # calculate neighboring arrays based on party centroid
                chase_space_points = np.array(self.party_centroids[candidate.party]) + np.array(chase_point)
                chase_space.append(chase_space_points)

            # Store all these possible opinion vectors in a dictionary, so we
            # can store this candidate's vote totals for each one.
            chase_space_tuples = [tuple(arr) for arr in chase_space]
            vote_counts = {t: 0 for t in chase_space_tuples}

            # Store the candidate's current (original) opinions, before making
            # any of these hypothetical choices.
            original_opinions = candidate.opinions
            for t in chase_space_tuples:
                candidate.opinions = t
                for voter in self.schedule.agents:
                    chosen_candidate = rational_vote(self, voter)
                    if chosen_candidate == candidate:
                        vote_counts[t] += 1

            # if the candidate got 0 votes, pick a random opinion array, given it is between 0-1
            if max(vote_counts.values()) == 0:
                self.datacollector.add_table_row("zero_votes",
                    { 'party': candidate.party,
                    'iter': self.step_num })
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

    # Compute the dimensionality-reduced version of the agent opinion matrix,
    # so that an approximation of it can be plotted in 2-dimensions.
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

    # Plot a 2-d approximation of the agents (in reduced opinion space).
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

    def make_anim(self, filename="CI2.gif"):
        os.system(f"convert -delay 25 -loop 0 output*.png {filename}")


# Represents a candidate of a given party, who has its own array of opinions.
class Candidate(mesa.Agent):
    def __init__(self, unique_id, model, opinions, party):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.party = party
    # Note: we can't have Candidates .step() themselves because they need to
    #   make their chase decisions synchronously, based on where each of them
    #   were in opinion space last election.

# represents a voter with an array of opinions.
# at each step of the simulation, voter x may influence voter y's
# opinion on an issue if they agree on a different issue.
class Voter(mesa.Agent):
    def __init__(self, unique_id, model, opinions, party, voting_algorithm):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.party = party
        self.voting_algorithm = voting_algorithm

    # If an agent's opinions have changed to be close enough to a different
    # party's centroid, switch that voter into the different party
    def switch_parties(self):
        min_distance = 100
        closest_party = -1
        for party in self.model.party_centroids:
            distance = math.sqrt(sum((x - y) ** 2 for x, y in zip(self.opinions, self.model.party_centroids[party])))
            if distance < min_distance:
                min_distance = distance
                closest_party = party
        if closest_party != self.party and min_distance < party_switch_threshold:
            self.model.datacollector.add_table_row("party_switches",
                { "agent_id": self.unique_id, "old_party": self.party,
                "new_party": closest_party, "iter": self.model.step_num})
            self.party = closest_party

    def step(self):
        # Implements the CI2 influence algorithm.
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
        # if x and y strongly disagree on i1, then x's opinion on i2 will move
        # further from y's opinion on i2
        elif diff >= pushaway:
            if self.opinions[i2] < neighbor.opinions[i2]:
                self.opinions[i2] -= ((self.opinions[i2]) / 2)
            else:
                self.opinions[i2] += ((1-self.opinions[i2])/ 2)
        # switch parties if party-switching criteria is met
        self.switch_parties()
        self.model.recompute_centroids()


def plot_election_outcomes(results):
    er = results['election_results']
    rr = results['rational_results']
    # Ugliest code ever? Candidate...
    er = pd.DataFrame.from_dict(dict(zip(er.index,er.values))).transpose()
    rr = pd.DataFrame.from_dict(dict(zip(rr.index,rr.values))).transpose()
    fig = plt.figure()
    axer = fig.add_subplot(211)
    axrr = fig.add_subplot(212)
    axer.title.set_text("Actual election results")
    axrr.title.set_text("Hypothetical (rational) results")
    axer.plot(er.index,er)
    axrr.plot(rr.index,rr)
    axer.set_ylim((0,er.max(axis=None)*1.1))
    axrr.set_ylim((0,er.max(axis=None)*1.1))
    axrr.set_xlabel(
        f"Election number (one per {election_steps} iterations)")
    plt.tight_layout()
    plt.savefig(f"election_outcomes.png")


def plot_party_switches(party_switches):
    plt.figure()
    ps_time = party_switches.value_counts('iter').sort_index()
    plt.plot(ps_time.index, ps_time)
    plt.title("Number of voter party switches")
    plt.xlabel("Simulation step")
    plt.ylabel("# voters who switched parties")
    plt.tight_layout()
    plt.savefig(f"party_switches.png")


# Parameters

# Threshold that determines when agents are close enough on one issue to
# assimilate on a second issue
openness = 0.1
# Threshold that determines when ages are far away enough on one issue to
# push away from each other on a second issue
pushaway = 0.6
# The number of opinions held by each voter and candidate
num_opinions = 3
# Number of nodes in the ER graph
N = 20
# Edge probability of the ER graph
edge_probability = 0.5
# Max number of the steps the simulation will run before terminating
max_iter = 400
# Threshold for how close voters' opinions need to be to a different party's
# centroid in order for them to switch to that party
party_switch_threshold = 0.2
# Threshold that determines how far away a voter needs to be from all
# candidates to not vote
no_vote_threshold = pushaway
# Number of candidates
num_candidates = 3
# Steps between each election
election_steps = 50
# Proportion of voters who will vote rationally
frac_rational = 0.7
# Proportion of voters who will vote for their party
frac_party = 0.1
# Proportion of voters who will vote using the "fast and frugal 1" algorithm
frac_ff1 = 0.1
# Proportion of voters who will vote using the "fast and frugal 2" algorithm
frac_ff2 = 0.1
# "Radius" of the hypercube in which candidates can move to chase votes
chase_radius = 0.2
if __name__ == "__main__":

    if len(sys.argv) <= 1:
        sys.exit("Usage: CI2elections.py numSims [animationFilename].")

    num_sims = int(sys.argv[1])
    if num_sims == 1 and len(sys.argv) == 3:
        do_anim = True
        anim_filename = sys.argv[1]
    else:
        do_anim = False

    params = {
        "N": N,
        "p": edge_probability,
        "party_switch_threshold": party_switch_threshold,
        "num_candidates": num_candidates,
        "max_iter": max_iter,  # only needed for plot caption
        "no_vote_threshold": no_vote_threshold,
        "frac_rational": frac_rational,
        "frac_party": frac_party,
        "frac_ff1": frac_ff1,
        "frac_ff2": frac_ff2,
        "chase_radius": chase_radius,
        "election_steps": election_steps
    }

    if num_sims == 1:
        # Single run.
        s = Society(params["N"], params["p"], params["party_switch_threshold"],
            params["num_candidates"], params["max_iter"],
            params["no_vote_threshold"], params["frac_rational"],
            params["frac_party"], params["frac_ff1"], params["frac_ff2"],
            params["chase_radius"], election_steps, do_anim)
        for i in range(max_iter):
            s.step()
        single_results = s.datacollector.get_model_vars_dataframe()
        party_switches = s.datacollector.get_table_dataframe("party_switches")
        zero_votes = s.datacollector.get_table_dataframe("zero_votes")
        if do_anim:
            print(f"Making animation {anim_filename}...")
            s.make_anim()

        # You now have single_results in your environment. For example, you
        # could do:
        # >>> single_results.iloc[0].election_results
        # to see the results at time=0.

        plot_election_outcomes(single_results)
        plot_party_switches(party_switches)
        if len(zero_votes) > 0:
            mins = zero_votes.groupby('party').min()
            for m in mins.itertuples():
                print(f"Candidate {m.Index} had zero votes as early as"
                    f" the step-{m.iter} election.")
        else:
            print("No zero vote candidates.")

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

