import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(len(df)))
num_periods = len(periods)
shift_length = 16
period_labels = df['Time'].tolist()
if df['Requirement'].isnull().any():
    raise ValueError("Missing values found in 'Requirement' column.")
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[p] for p in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = [p for p in periods if (t - p) % num_periods < shift_length]
    m.addConstr(gp.quicksum((x[p] for p in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x[p].X)) for p in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Schedule ---')
    for p in periods:
        val = int(round(x[p].X))
        if val > 0:
            print(f'Start at period {p:2d} ({period_labels[p]}): {val} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')