#!/usr/bin/env python3

from CI2elections import plot_rationality_over_time, plot_winners_over_time
from CI2elections import plot_chase_dists, PLOT_DIR

import argparse
import pandas as pd
import os

def hydrate_sim_args(sim_tag):
    sim_args = {}
    with open(os.path.join(PLOT_DIR, f"{sim_tag}_args.txt")) as f:
        for line in [ l.strip() for l in f.readlines() ]:
            param, val = line.split(" = ")
            if val.isnumeric():
                val = int(val)
            elif val.replace(".","").isnumeric():
                val = float(val)
            elif "[" in val:
                val = eval(val)
            sim_args[param] = val
    return sim_args


parser = argparse.ArgumentParser(description="Election ABM plot re-generator.")
parser.add_argument("sim_tag", type=str,
    help="A string previously used to run sims, whose plots to recreate.")

plot_args = parser.parse_args()

print("Loading sim parameters...")
sim_args = hydrate_sim_args(plot_args.sim_tag)
sweep_vars = { var : val for var, val in sim_args.items()
    if type(val) is list  and  len(val) > 1 }
args = argparse.Namespace(**sim_args)
globals()['args'] = args

print("Loading data...")
for t in ['er','rr','cd']:
    globals()[t] = pd.read_csv(os.path.join(PLOT_DIR,
        f"{plot_args.sim_tag}_{t}.csv"), index_col=None)
    globals()[t].columns = [ int(c) if c.isnumeric() else c
        for c in globals()[t].columns ]

print("Regenerating plots...")
plot_rationality_over_time(er, rr, list(sweep_vars.keys()), args)
plot_winners_over_time(er, list(sweep_vars.keys()), args)
plot_chase_dists(cd, list(sweep_vars.keys()), args)
