import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_periods = 48
shift_length = 16
if df.shape[0] != num_periods:
    raise ValueError(f'Expected {num_periods} periods, found {df.shape[0]} rows in CSV.')
requirement = df['Requirement'].astype(int).to_dict()
periods = list(range(num_periods))
shift_starts = list(range(num_periods))
covering_shifts = {t: [s for s in shift_starts if (t - s) % num_periods < shift_length] for t in periods}
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts[t])) >= requirement[t], name=f'cov_{t}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal}')
    for s in shift_starts:
        print(f'x[{s}] {x_vars[s].VarName} {x_vars[s].X}')
else:
    print(f'Solver status: {m.status}')