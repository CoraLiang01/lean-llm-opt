import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirements = df['Requirement'].astype(int).to_dict()
shift_starts = periods
shift_covers = dict()
for s in shift_starts:
    covered = [(s + i) % n_periods for i in range(16)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in periods}
for s in shift_starts:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)

def solve_shift_scheduling():
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirements[t], name=f'cover_{t}')
    m.optimize()
    return m
m = solve_shift_scheduling()