Abstract (SD)
=============
* an ABM of the democratic election campaign process 
* we focus on issue-based politics (not, say, emotional messaging) which seems missing from the literature
* voters influence one another, and candidates adjust positions to "chase" them
* voters use various algorithms to decide whether to vote and who for
* allows us to experiment and qualitatively answer questions like:
    * how much do candidates drift? do high-drifting ones outperform low?
    * how does the electorate's composition of voter strategies affect how often election outcomes are "rational?"


Introduction (SD)
=================
In real life, politically-minded people have opinions on multiple issues.
Whether in person or online, they interact on a social network and influence
each other. Homophily suggests something like the CI2 dynamic.

[Disclaimer: yes, there are of course innumerable other factors that go into
one's political opinions and voting decisions, including forms of mass media,
stable factors like demographics/economics/geography, etc. We can't model
everything, and in this paper we're focusing on issue-based voting.]

[Disclaimer: we're also not modeling turnout]

(Maybe include this: also in real life, not everyone feels equally strongly
about every issue.)

Political parties have platform positions, explicitly or implicitly, which can
attract or repel voters. At least at some level, a voter chooses a
candidate/party who aligns with them on important issues. Candidates and
parties lean into issues that they perceive to be "wins" for them; and shy away
from, or even change their position on, those they think will lose them votes.
In other words, they maneuver in issue space to maximize votes.

The simulation we present in this paper models this process of voters and
candidates altering their viewpoints over time in response to each other.



Related Work
============
1. Background
    1. Homophily as a near-universal driver of human behavior **(SD)**
    2. Factors that go into voting **(HP)**
        * choice of candidate
        * whether to "turn out"

2. Competitors
    1. Lots of OD lit, but remarkably little on continuous opinion vectors **(SD)**
    2. Other OD approaches to continuous opinion vectors **(SD)**
    3. Approaches to actually modeling the election process **(HP)**
        * note that this is different than election forecasting


The Model
=========
1. Overall description: voters and candidates interact in opinion space **(SD)**
2. Voter agents **(SD)**
    1. Modeling opinions on issues as continuous-valued vectors
    2. How voters choose who to interact with
    3. What happens at interaction -- CI2
    4. Turnout: an voter's decision about whether or not to vote
    5. Different voting algorithms **(HP)**
        * Rational
        * Bounded Rational (or "Constrained Rational")
        * F&F1
        * F&F2
        * Party-line vote
3. Parties **(HP)**
    1. How these are initialized, both for voters and candidates
    2. Whether and how voters ever change parties
4. Candidate agents **(HP)**
    1. Their issue positions are modeled similar to voter opinions
    2. The "drift" algorithm
5. Elections **(HP)**
    1. At regular points in time, elections are held, and results tabulated


Experimentation
===============
1. Verification: sanity-checking that the model works as designed **(SD)**
2. Our independent variables **(HP)**
    1. What each i.v. is, and how it is measured
    2. The range of values we sweep each one through
3. Our dependent variables **(SD)**
    1. What each d.v. is, and how it is measured
    2. Initial hypotheses about how each d.v. will react to the i.v.'s


Results
=======
1. Representative plots, and narrative, from single simulations **(SD)**
1. Plots, and narrative, from parameter sweeps **(HP)**


Discussion
==========
1. General findings about candidate "drift." Do candidates tend to drift the
same amount? Do high-drifters tend to do better in elections than low-drifters?
Etc. **(SD)**
2. General findings about voter algorithm mix. How much does this matter? How
often are elections rational based on various compositions? Do certain voter
algorithms make drifting less/more effective? **(HP)**


Future work
===========
1. Third parties: under what circumstances is the model more conducive to other
parties arising? **(HP)**
2. If campaigns can microtarget which voters are likely to use which voting
algorithms, can they use that to their advantage in drifting more
strategically? **(SD)**


Conclusions
===========


Refs
====
1. Link to github repo, and encouragement to view/use/contribute **(HP)**
2. Links to animations from sample simulation runs **(SD)**

Bibliography
============
