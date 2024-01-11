#!/usr/bin/env python3

import mesa
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colormaps
import numpy as np
import pandas as pd
import math
import sys
import random
import os
import argparse
from copy import copy
from sklearn.preprocessing import StandardScaler
from mesa.time import RandomActivation
from mesa.batchrunner import batch_run
from mesa.datacollection import DataCollector
from itertools import product


def dist(x, y):
    """
    Return the Euclidean distance between two points, represented as NumPy
    arrays.
    """
    assert type(x) is np.ndarray
    assert type(y) is np.ndarray
    return np.sqrt(((x - y)**2).sum())

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
def rational_vote(society, voter):
    min_distance = 10
    for candidate in society.candidates:
        distance = dist(voter.opinions, candidate.opinions)
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    return closest_candidate

# Returns the candidate number that belongs to the same party as the voter
def party_vote(society, voter):
    for candidate in society.candidates:
        if voter.party == candidate.party:
            return candidate

def ff1_vote(society, voter):
    min_distance = 10
    for candidate in society.candidates:
        distance = abs(candidate.opinions[voter.ff1_issue]
                       - voter.opinions[voter.ff1_issue])
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    return closest_candidate

def ff2_vote(society, voter):
    min_distance = 10
    for candidate in society.candidates:
        distance = abs(candidate.opinions[society.ff2_issue]
                       - voter.opinions[society.ff2_issue])
        if distance < min_distance:
            min_distance = distance
            closest_candidate = candidate
    return closest_candidate

# An agent-based model of a society with N voters and num_candidates political
# candidates for office. Other constructor parameters:
# p - probability of social connection between any two voters
# party_switch_threshold - threshold for how close voters' opinions need to be
#    to a different party's centroid in order for them to switch to that party
# num_candidates - number of candidates in the election
# num_opinions - number of issues that voters/candidates have opinions on
# max_iter - the longest time the sim will run before being terminated
# frac_rational -- the proportion of voters who will vote rationally
# frac_party -- the proportion of voters who will vote based solely on party
# frac_ff1 -- the proportion of voters who will vote using the "fast & frugal"
#    algorithm, variant 1
# frac_ff2 -- the proportion of voters who will vote using the "fast & frugal"
#    algorithm, variant 2
# chase_radius -- "Radius" of the hypercube in which candidates can chase votes
# election_steps -- hold an election every this number of steps
# do_anim -- create single-sim animation?
class Society(mesa.Model):
    def __init__(self, args, do_anim=False):

        super().__init__()
        self.__dict__.update(vars(args))
        self.graph = gen_graph(self.N, self.edge_probability)

        # If N is 50 and num_candidates is 3, then agents 0, 1, etc are nodes
        # 0, 1, ... 49 in the graph, and candidates 0, 1, 2 are 50, 51, 52.
        for c in range(self.N, self.N + self.num_candidates):
            self.graph.add_node(c)

        self.pos = nx.spring_layout(self.graph)
        self.schedule = RandomActivation(self)
        self.candidates = []
        self.voters = []
        self.step_num = 0
        self.party_centroids = {}
        self.ff2_issue = np.random.randint(self.num_opinions)
        assert math.isclose(self.frac_rational + self.frac_party +
            self.frac_ff1 + self.frac_ff2, 1.0), \
            (f"Electorate of {self.frac_rational}, {self.frac_party}, " +
            f"{self.frac_ff1}, {self.frac_ff2} does not add up to 1.0.")

        self.do_anim = do_anim

        # Create new random candidates, one for each party, and initialize each
        # party's "centroid" to be not actually its centroid of voters, but its
        # candidate's opinion vector. (TODO Issue #1)
        for i in range(self.num_candidates):
            newCandidate = Candidate(i, self,
                np.random.uniform(0,1,self.num_opinions), i,
                self.chase_radius if i < self.num_chasers else 0)
            self.party_centroids[i] = newCandidate.opinions
            self.candidates.append(newCandidate)

        # Create random voters, and assign each one to the party whose centroid
        # (candidate; see above) it is closest to in Euclidean opinion space.
        for i in range(self.N):
            newVoter = Voter(i, self,
                np.random.uniform(0,1,self.num_opinions), 0, rational_vote,
                np.random.randint(self.num_opinions))
            min_distance = 100
            closest_party = -1
            for party in self.party_centroids:
                distance = dist(newVoter.opinions, self.party_centroids[party])
                if distance < min_distance:
                    min_distance = distance
                    closest_party = party
            newVoter.party = closest_party
            self.voters.append(newVoter)
            self.schedule.add(newVoter)
        self.determine_voting_algorithms()
        self.datacollector = DataCollector(
            agent_reporters={},
            model_reporters={"rational_results": Society.rational_elect,
                             "election_results": Society.elect,
                             "chase_distances": Society.get_chase_dists},
            tables={
                "party_switches": ["agent_id","old_party","new_party","iter"],
                "zero_votes": ["party","iter"]
            }
        )

    # Method used at initialization to determine all voters' voting algorithms
    # based on fractions specified.
    def determine_voting_algorithms(self):
        voters = copy(self.schedule.agents)
        num_rational = int(self.frac_rational * self.N)
        num_party = int(self.frac_party * self.N)
        num_ff1 = int(self.frac_ff1 * self.N)
        rational_voters = random.sample(voters, num_rational)
        voters = [voter for voter in voters if voter not in rational_voters]
        party_voters = random.sample(voters, num_party)
        voters = [voter for voter in voters if voter not in party_voters]
        ff1_voters = random.sample(voters, num_ff1)
        for voter in self.schedule.agents:
            if voter in rational_voters:
                voter.voting_algorithm = rational_vote
            elif voter in party_voters:
                voter.voting_algorithm = party_vote
            elif voter in ff1_voters:
                voter.voting_algorithm = ff1_vote
            else:
                voter.voting_algorithm = ff2_vote

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
            new_centroid = np.zeros(self.num_opinions)
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


    # Run an election, if it's time to. (This function will simply return
    # all zeros if it's not an election time step.)
    # Runnin an election involves having each agent vote according to their
    # own algorithm (e.g., party-based, rational) and also vote rationally
    # (regardless of algorithm) so we have election results on each. Then,
    # after votes are tabulated, have each candidate strategically chase()
    # towards a new opinion vector that will get them the most votes.
    #
    # Returns a list of vote counts, ordered by candidate number.
    def elect(self):
        self.ff2_issue = np.random.randint(self.num_opinions)
        real_vote_counts = {candidate.unique_id: 0 for candidate in self.candidates}
        if self.step_num % self.election_steps != 0:
            # Not time to run an election. Go back to sleep.
            return list(real_vote_counts.values())

        # Have all agents vote based on their voting algorithm and store the
        # vote counts in real_vote_counts
        for voter in self.schedule.agents:
            chosen_candidate = voter.voting_algorithm(self, voter)
            real_vote_counts[chosen_candidate.unique_id] += 1

        # chase after election
        new_opinions = self.chase()

        # Store the chase distances in an inst var, for retrieval in later
        # call to model reporter "get_chase_dists()"; any other way seems
        # clunkier.  -SD
        self.chase_dists = {}
        for candidate in self.candidates:
            self.chase_dists[candidate.party] = \
                dist(candidate.opinions, new_opinions[candidate.party])
            candidate.opinions = new_opinions[candidate.unique_id]
        return np.array(list(real_vote_counts.values()))

    def get_chase_dists(self):
        # These should have been computed by the immediately preceding call
        # to .elect() (via the "election_results" model reporter.)
        if self.step_num % self.election_steps != 0:
            # Not time to run an election. Go back to sleep.
            return [0] * len(self.candidates)
        return list(self.chase_dists.values())

    # See comments on .elect(). All is the same, except that .rational_elect()
    # disregards voting algorithm, always using rational_vote() instead. Also,
    # no candidate chasing is performed.
    def rational_elect(self):
        rational_vote_counts = {candidate.unique_id: 0 for candidate in self.candidates}
        if self.step_num % self.election_steps != 0:
            # Not time to run an election. Go back to sleep.
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
            num_points = int(candidate.chase_radius * 40) + 1
            chase_spectrum = np.linspace((-1 * candidate.chase_radius),
                candidate.chase_radius, num_points)

            # Generate the Cartesian product of num_opinions copies of the
            # chase_spectrum. This gives us an iterator of tuples, each of size
            # num_opinions, that represents a possible delta-shift (in opinion
            # space) that this candidate will consider moving to in order to
            # get more votes.
            chase_deltas = product(chase_spectrum,
                repeat=len(candidate.opinions))
            chase_space_points = [ tuple(np.array(candidate.opinions) + delta)
                for delta in chase_deltas ]

            # Store all these possible opinion vectors in a dictionary, so we
            # can store this candidate's vote totals for each one.
            vote_counts = {p: 0 for p in chase_space_points}

            # Store the candidate's current (original) opinions, before making
            # any of these hypothetical choices.
            original_opinions = copy(candidate.opinions)
            for p in chase_space_points:
                candidate.opinions = np.array(p)
                for voter in self.schedule.agents:
                    chosen_candidate = rational_vote(self, voter)
                    if chosen_candidate == candidate:
                        vote_counts[p] += 1

            # if the candidate got 0 votes no matter where it probed, just
            # a random one to move to.
            if max(vote_counts.values()) == 0:
                self.datacollector.add_table_row("zero_votes",
                    { 'party': candidate.party,
                    'iter': self.step_num })
                best_chase_point = np.array(list(vote_counts.keys()))[
                    np.random.randint(len(vote_counts))]

            # if the candidate got more than 0 votes, retrieve the key (opinions) corresponding
            # to the max vote count. if any of the opinions are outside the 0-1 range, clip
            # them back to 0 or 1
            else:
                best_chase_point = np.array(max(vote_counts,
                    key=lambda k: vote_counts[k])).clip(0,1)

            # store the optimal opinions and set the candidate's opinions back to
            # what they were originally
            candidate.opinions = original_opinions
            optimal_opinions[candidate.unique_id] = best_chase_point
        return optimal_opinions

    # Compute the dimensionality-reduced version of the agent opinion matrix,
    # so that an approximation of it can be plotted in 2-dimensions.
    def compute_SVD(self):
        X = np.r_[[ a.opinions for a in self.schedule.agents ],
            [ c.opinions for c in self.candidates ]]
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
        cmap = colormaps['Set1'].colors
        plt.clf()
        apositions = {a:(svecs[a,0],svecs[a,1]) for a in range(self.N)}
        cpositions = {c:(svecs[c,0],svecs[c,1]) for c in range(self.N, self.N +
            self.num_candidates)}
        positions = copy(apositions)
        positions.update(cpositions)
        if self.color_nodes == "opinion":
            acolors = [ a.opinions for a in self.schedule.agents ]
            ccolors = [[1,1,.5]]*self.num_candidates
        elif self.color_nodes == "wouldVoteFor":
            acolors = [ cmap[a.voting_algorithm(self,a).party]
                for a in self.schedule.agents ]
            ccolors = [ cmap[c.party] for c in self.candidates ]
        else:
            sys.exit("Invalid node coloring scheme.")
        colors = acolors + ccolors
        labels = { a:str(a) for a in range(self.N) }
        labels.update({ self.N + c:str(c) for c in range(self.num_candidates) })
        nx.draw_networkx_edges(self.graph, positions)
        nx.draw_networkx_nodes(self.graph, apositions, nodelist=range(self.N),
            node_color=acolors, node_shape="o", node_size=250)
        nx.draw_networkx_nodes(self.graph, cpositions, nodelist=range(self.N,
            self.N + self.num_candidates), node_color=ccolors,
            edgecolors=[.2,.2,.2],node_shape="*", node_size=700)
        nx.draw_networkx_labels(self.graph, positions, labels=labels, font_size=10)
        plt.xlim((-1.2,1.2))
        plt.ylim((-1.2,1.2))
        plt.title(f"Time {self.step_num} of {self.max_iter}")
        plt.savefig(f"{self.sim_tag}_output{self.step_num:03}.png", dpi=300)
        plt.close()

    def make_anim(self, filename="CI2.gif"):
        if not filename.endswith(".gif"):
            filename += ".gif"
        os.system(
            f"convert -delay 25 -loop 0 {self.sim_tag}_output*.png {filename}")


# Represents a candidate of a given party, who has its own array of opinions.
class Candidate(mesa.Agent):
    def __init__(self, unique_id, model, opinions, party, chase_radius):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.party = party
        self.chase_radius = chase_radius
    # Note: we can't have Candidates .step() themselves because they need to
    #   make their chase decisions synchronously, based on where each of them
    #   were in opinion space last election.

# represents a voter with an array of opinions.
# at each step of the simulation, voter x may influence voter y's
# opinion on an issue if they agree on a different issue.
class Voter(mesa.Agent):
    def __init__(self, unique_id, model, opinions, party, voting_algorithm,
                 ff1_issue):
        super().__init__(unique_id, model)
        self.opinions = opinions
        self.party = party
        self.voting_algorithm = voting_algorithm
        self.ff1_issue = ff1_issue
    # If an agent's opinions have changed to be close enough to a different
    # party's centroid, switch that voter into the different party
    def switch_parties(self):
        min_distance = 100
        closest_party = -1
        for party in self.model.party_centroids:
            distance = dist(self.opinions, self.model.party_centroids[party])
            if distance < min_distance:
                min_distance = distance
                closest_party = party
        if (closest_party != self.party and
            min_distance < self.model.party_switch_threshold):
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
        i1 = np.random.choice(np.arange(self.model.num_opinions))
        diff = abs(self.opinions[i1] - neighbor.opinions[i1])
        # chooses another random issue, i2, that is different from i1
        i2 = np.random.choice(np.arange(self.model.num_opinions))
        while i2 == i1:
            i2 = np.random.choice(np.arange(self.model.num_opinions))
        # if x and y agree on i1, then x's opinion on i2 will move
        # closer to y's opinion on i2
        if diff <= self.model.openness:
            self.opinions[i2] = (self.opinions[i2] + neighbor.opinions[i2])/2
        # if x and y strongly disagree on i1, then x's opinion on i2 will move
        # further from y's opinion on i2
        elif diff >= self.model.pushaway:
            if self.opinions[i2] < neighbor.opinions[i2]:
                self.opinions[i2] -= ((self.opinions[i2]) / 2)
            else:
                self.opinions[i2] += ((1-self.opinions[i2])/ 2)
        # switch parties if party-switching criteria is met
        self.switch_parties()
        self.model.recompute_centroids()


def get_election_results(results):
    """
    Given a results DataFrame from a single or batch run, produce three
    DataFrames of results, with one row per election number: candidate vote
    totals for actual elections, candidate vote totals for rational elections,
    and candidate chase distances.

    Input: for single runs, results should look like: (same for chase dists)
        rational_results election_results
    0          [0, 0, 0]        [0, 0, 0]    # <- all 0's because no election
    1          [0, 0, 0]        [0, 0, 0]    # was run at any of these times
    2          [0, 0, 0]        [0, 0, 0]
    ...
    49        [19, 0, 1]       [16, 4, 0]    # <- ah! an actual election
    ...

    For batch runs, results should look like: (same for chase dists)
        RunId   rational_results election_results
    0       0          [0, 0, 0]        [0, 0, 0]
    1       0          [0, 0, 0]        [0, 0, 0]
    2       0          [0, 0, 0]        [0, 0, 0]
    ...
    49     19         [19, 0, 1]       [16, 4, 0]
    ...


    Output: for single runs, each of the three DataFrames will look like this:
        0   1   2
    0  16   4   0
    1   1  17   2
    2   3  17   0
    ...
    """
    # Ugliest code ever? Candidate...
    er = results['election_results']
    rr = results['rational_results']
    cd = results['chase_distances']
    er = pd.DataFrame.from_dict(dict(zip(er.index,er.values))).transpose()
    rr = pd.DataFrame.from_dict(dict(zip(rr.index,rr.values))).transpose()
    cd = pd.DataFrame.from_dict(dict(zip(cd.index,cd.values))).transpose()
    election_times = er.sum(axis=1) > 0
    if 'RunId' in results:
        # Batch run
        runId_step = results[['RunId','Step']]
        runId_step.loc[:,'Step'] += 1   # this is a sin, aligning iterations
                                        # this way, and I will pay for it in
                                        # the afterlife
        er = pd.concat([runId_step,er],axis=1)
        rr = pd.concat([runId_step,rr],axis=1)
        cd = pd.concat([runId_step,cd],axis=1)
    er = er[election_times].reset_index(drop=True)
    rr = rr[election_times].reset_index(drop=True)
    cd = cd[election_times].reset_index(drop=True)
    if 'RunId' in results:
        # Batch run
        elec_num = er.Step / er.Step.min()
        elec_num = elec_num.astype(int)
        er['elec_num'] = elec_num
        rr['elec_num'] = elec_num
        cd['elec_num'] = elec_num
    return er, rr, cd

def plot_election_outcomes(results, sim_tag):
    # Single run
    fig = plt.figure()
    er, rr, _ = get_election_results(results)
    axer = fig.add_subplot(211)
    axrr = fig.add_subplot(212)
    axer.title.set_text("Actual election results")
    axrr.title.set_text("Hypothetical (rational) results")
    axer.plot(er.index,er)
    axrr.plot(rr.index,rr)
    axer.set_ylim((0,er.max(axis=None)*1.1))
    axrr.set_ylim((0,er.max(axis=None)*1.1))
    axrr.set_xlabel(
        f"Election number (one per {len(results) // len(er)} iterations)")
    plt.tight_layout()
    plt.savefig(f"{sim_tag}_election_outcomes.png", dpi=300)
    plt.close()


def plot_party_switches(party_switches, sim_tag):
    # Single run
    plt.figure()
    ps_time = party_switches.value_counts('iter').sort_index()
    plt.plot(ps_time.index, ps_time)
    plt.title("Number of voter party switches")
    plt.xlabel("Simulation step")
    plt.ylabel("# voters who switched parties")
    plt.tight_layout()
    plt.savefig(f"{sim_tag}_party_switches.png", dpi=300)
    plt.close()


def plot_rationality_over_time(batch_results):
    # Batch run
    plt.figure()
    er, rr, _ = get_election_results(batch_results)
    compute_winners(er, args.num_candidates)
    compute_winners(rr, args.num_candidates)
    er['rational'] = er.winner == rr.winner
    frac_rational_by_elec_num = (er[['elec_num','rational']].groupby(
        'elec_num').mean('rational') * 1).rational
    # Plot error bars to 95% confidence interval
    ci = 1.96 * np.sqrt(frac_rational_by_elec_num *
        (1 - frac_rational_by_elec_num) / len(frac_rational_by_elec_num))
    frac_rational_by_elec_num.plot(kind='bar',
        yerr=np.c_[np.minimum(frac_rational_by_elec_num,ci),
        np.minimum(ci,1-frac_rational_by_elec_num)].transpose(),
        capsize=5)
    plt.ylim((0,1.1))
    plt.title(f"(Elections every {args.election_steps} steps)")
    if args.sim_tag:
        plt.suptitle(f"% rational election outcomes -- {args.sim_tag}")
    else:
        plt.suptitle(f"% rational election outcomes")
    plt.savefig(f'{args.sim_tag}_fracRational.png', dpi=300)
    plt.close()

def plot_winners_over_time(batch_results):
    # Batch run
    plt.figure()
    er, _, _ = get_election_results(batch_results)
    compute_winners(er, args.num_candidates)
    cand_wins = er.groupby('elec_num').winner.value_counts()
    cand_wins = pd.DataFrame(cand_wins).reset_index()
    num_voters = cand_wins[cand_wins.elec_num==1]['count'].sum()
    cand_wins['% wins'] = cand_wins['count'] / num_voters * 100
    sns.catplot(x="elec_num", y="% wins", hue="winner",
        data=cand_wins, kind="bar", palette="Set1")
    plt.ylim((0,min(cand_wins['% wins'].max()+10,110)))
    if args.sim_tag:
        plt.suptitle(f"% election wins by candidate -- {args.sim_tag}")
    else:
        plt.suptitle(f"% election wins by candidate")
    #plt.tight_layout()
    plt.savefig(f'{args.sim_tag}_winners.png', dpi=300)
    plt.close()

def plot_chase_dists(batch_results):
    # Batch run
    plt.figure()
    _, _, cd = get_election_results(batch_results)
    cols = {}
    for party in range(args.num_candidates):
        line_title = f'Candidate {party} '
        line_title += "(chaser)" if party < args.num_chasers else "(non)"
        cols[line_title] = cd.groupby('elec_num')[party].mean()
    chase_dists = pd.DataFrame(cols)
    chase_dists.plot(kind="line")
    if args.sim_tag:
        plt.suptitle(f"Mean candidate chase distance by election -- {args.sim_tag}")
    else:
        plt.suptitle(f"Mean candidate chase distance by election")
    ##plt.tight_layout()
    plt.savefig(f'{args.sim_tag}_chase_dists.png', dpi=300)
    plt.close()

def compute_winners(results, num_candidates):
    """
    Given a DataFrame that has, possibly among other columns, integer-named
    columns containing vote counts, add to it a new column called "winner"
    with the candidate/party number who won each election.
    """
    assert num_candidates-1 in results.columns
    assert num_candidates not in results.columns

    winners = results[range(num_candidates)].idxmax(axis=1)
    results['winner'] = winners


parser = argparse.ArgumentParser(description="Election ABM.")
parser.add_argument("-n", "--num_sims", type=int, default=1,
    help="Number of simulations to run (1 = single mode, >1 = batch mode)")
parser.add_argument("--openness", type=float, default=0.1,
    help="Threshold when agents are close enough on one issue to attract "
        "on a second issue")
parser.add_argument("--pushaway", type=float, default=0.6,
    help="Threshold when agents are far enough away on one issue to push "
        "away on a second issue")
parser.add_argument("--num_opinions", type=int, default=3,
    help="Number of opinions held by each voter and candidate")
parser.add_argument("-N", type=int, default=20,
    help="Number of voter agents (and nodes in the ER graph)")
parser.add_argument("--num_candidates", type=int, default=3,
    help="Number of candidates")
parser.add_argument("--edge_probability", type=float, default=0.5,
    help="Edge probability of the ER graph")
parser.add_argument("--max_iter", type=int, default=400,
    help="Max number of the steps the simulation will run before terminating")
parser.add_argument("--party_switch_threshold", type=float, default=0.2,
    help="Threshold for how close voters' opinions need to be to a different "
        "party's # centroid in order for them to switch to that party")
parser.add_argument("--election_steps", type=int, default=50,
    help="Steps between each election")
parser.add_argument("--frac_rational", type=float, default=0.7,
    help="Proportion of voters who will vote rationally")
parser.add_argument("--frac_party", type=float, default=0.1,
    help="Proportion of voters who will vote solely based on party")
parser.add_argument("--frac_ff1", type=float, default=0.1,
    help="Proportion of voters who will use the 'fast & frugal 1' voting alg")
parser.add_argument("--frac_ff2", type=float, default=0.1,
    help="Proportion of voters who will use the 'fast & frugal 2' voting alg")
parser.add_argument("--chase_radius", type=float, default=0.2,
    help="'Radius' of the hypercube in which candidates can chase votes")
parser.add_argument("--num_chasers", type=float, default=sys.maxsize,
    help="The number of candidates who will chase voters (others stay put)")
parser.add_argument("--animation_filename", type=str, default=None,
    help="Filename in which to store single-sim animation (if any)")
parser.add_argument("--color_nodes", type=str, default="opinion",
    choices=["opinion","wouldVoteFor"],
    help="How to color nodes in single-sim animation (if any)")
parser.add_argument("--sim_tag", type=str, default=None,
    help="A string to use as prefix to plots produced (no spaces please)")



if __name__ == "__main__":

    args = parser.parse_args()

    np.set_printoptions(precision=4)

    do_anim = (args.num_sims == 1 and args.animation_filename)

    if args.num_sims == 1:
        # Single run.
        s = Society(args, do_anim)
        for i in range(args.max_iter):
            s.step()
        single_results = s.datacollector.get_model_vars_dataframe()
        party_switches = s.datacollector.get_table_dataframe("party_switches")
        zero_votes = s.datacollector.get_table_dataframe("zero_votes")
        if do_anim:
            print(f"Building animation {args.animation_filename}...")
            s.make_anim(args.animation_filename)
            print(f"...done.")

        # You now have single_results in your environment. For example, you
        # could do:
        # >>> single_results.iloc[0].election_results
        # to see the results at time=0.

        plot_election_outcomes(single_results, args.sim_tag)
        plot_party_switches(party_switches, args.sim_tag)
        if len(zero_votes) > 0:
            mins = zero_votes.groupby('party').min()
            for m in mins.itertuples():
                print(f"Candidate {m.Index} had zero votes as early as"
                    f" the step-{m.iter} election.")
        else:
            print("No zero vote candidates.")

    else:

        batch_results = batch_run(Society,
            parameters={ 'args': args },
            iterations=args.num_sims,
            max_steps=args.max_iter,
            number_processes=None,     # make this 1 to use only one CPU
            data_collection_period=1   # grab data every step
        )

        batch_results = pd.DataFrame(batch_results)
        er, rr, cd = get_election_results(batch_results)

        # You now have batch_results in your environment. For example, you
        # could do:
        # >>> batch_results.iloc[0].election_results
        # to see the results of the first simulation in the suite, at time=0.
        # (See other columns in batch_results to explain what each line
        # signifies.)
        # As a bonus, you also have er, rr, and cd in your environment, which
        # gives you the vote totals for all elections in all the batch runs,
        # and the chase distances.

        plot_rationality_over_time(batch_results)
        plot_winners_over_time(batch_results)
        plot_chase_dists(batch_results)
