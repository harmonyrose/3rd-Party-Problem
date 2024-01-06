#/usr/bin/env bash
# Recommend to run with "bash -x runs.sh" to see progressive output.

echo Running examples...

./CI2elections.py -n 50 -N 50 --frac_rational 1 --frac_party 0 --frac_ff1 0 --frac_ff2 0 --sim_tag "r1_p0_ff10_ff20"
./CI2elections.py -n 50 -N 50 --frac_rational 0 --frac_party 1 --frac_ff1 0 --frac_ff2 0 --sim_tag "r0_p1_ff10_ff20"
./CI2elections.py -n 50 -N 50 --frac_rational 0 --frac_party 0 --frac_ff1 1 --frac_ff2 0 --sim_tag "r0_p0_ff11_ff20"
./CI2elections.py -n 50 -N 50 --frac_rational 0 --frac_party 0 --frac_ff1 0 --frac_ff2 1 --sim_tag "r0_p0_ff10_ff21"
./CI2elections.py -n 50 -N 50 --frac_rational .5 --frac_party .5 --frac_ff1 0 --frac_ff2 0 --sim_tag "r.5_p.5_ff10_ff20"


# vim:textwidth=99999
