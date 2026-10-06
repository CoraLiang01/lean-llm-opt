import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv', sep=',')
shifts = df['Shift'].astype(int).tolist()
n_periods = len(shifts)
if n_periods != 24:
    raise ValueError(f'Expected 24 shifts, got {n_periods}')
required = dict(zip(df['Shift'].astype(int), df['Number Required'].astype(int)))

def solve_problem():
    m = gp.Model('BusCrewScheduling')
    x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in shifts)), gp.GRB.MINIMIZE)
    for h in shifts:
        covered_starts = [(h - i - 1) % 24 + 1 for i in range(4)]
        m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= required[h], name=f'cover_{h}')
    m.optimize()
    return m
m = solve_problem()