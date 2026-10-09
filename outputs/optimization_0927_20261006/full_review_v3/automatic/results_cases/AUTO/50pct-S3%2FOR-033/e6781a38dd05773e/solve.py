import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_slots = df['Time'].tolist()
if len(time_slots) != 48 or len(set(time_slots)) != 48:
    raise ValueError("Expected 48 unique time slots in 'Time' column.")
df['Requirement'] = df['Requirement'].astype(int)
requirement = dict(zip(time_slots, df['Requirement']))
shift_length = 16
num_slots = len(time_slots)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(time_slots, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in time_slots)), gp.GRB.MINIMIZE)
for (t_idx, t) in enumerate(time_slots):
    covering_starts = []
    for (s_idx, s) in enumerate(time_slots):
        rel = (t_idx - s_idx) % num_slots
        if 0 <= rel < shift_length:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift starts cover time slot {t} (index {t_idx})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t_idx}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in time_slots:
        val = x_vars[s].X
        if val > 1e-06:
            print(f"  Shift starting at '{s}': {int(round(val))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')