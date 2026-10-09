import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df.index)
period_labels = df['Time'].tolist()
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
periods_per_shift = 16
num_periods = len(period_ids)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
shift_start_ids = period_ids.copy()
covering_shifts = {i: [s for s in shift_start_ids if (i - s) % num_periods < periods_per_shift] for i in period_ids}
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_ids)), gp.GRB.MINIMIZE)
for i in period_ids:
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts[i])) >= requirement[i], name=f'cover_{i}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((x_vars[s].X for s in shift_start_ids))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_ids:
        val = x_vars[s].X
        if val > 1e-06:
            print(f"  Start at '{period_labels[s]}': {int(round(val))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')