import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_labels = df['Time'].tolist()
requirement = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in periods:
    covering_starts = [s for s in periods if (t - s) % n_periods < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum number of waitstaff)')
    print('--- Shift Start Assignments ---')
    for s in periods:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'Start at period {s} ({period_labels[s]}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')