import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
requirement = df['Requirement'].astype(int).to_dict()

def solve_shift_scheduling():
    m = gp.Model('WaitstaffShiftScheduling')
    x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
    coverage = {t: [] for t in periods}
    for s in periods:
        covered = [(s + offset) % n_periods for offset in range(16)]
        for t in covered:
            coverage[t].append(s)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in coverage[t])) >= requirement[t], name=f'cover_{t}')
    m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_shift_scheduling()