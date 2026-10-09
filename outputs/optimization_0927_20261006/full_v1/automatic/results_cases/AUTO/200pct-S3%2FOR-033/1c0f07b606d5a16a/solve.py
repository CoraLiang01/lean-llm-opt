import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Requirement' not in df.columns:
    raise KeyError("Column 'Requirement' not found in the CSV.")
requirements = df['Requirement'].astype(int).to_numpy()
num_slots = len(requirements)
if num_slots != 48:
    raise ValueError(f'Expected 48 time slots, got {num_slots}.')
time_slots = list(range(num_slots))
shift_starts = list(range(num_slots))
shift_length = 16
covering_shifts = {t: [] for t in time_slots}
for s in shift_starts:
    for i in range(shift_length):
        t = (s + i) % num_slots
        covering_shifts[t].append(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in time_slots:
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule (nonzero only) ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.iloc[s]['Time']
            print(f'Shift start at slot {s} ({time_label}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')