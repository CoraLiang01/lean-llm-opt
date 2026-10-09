import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {periods[i]: i for i in range(48)}
try:
    requirement = [int(x) for x in df['Requirement']]
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
if len(requirement) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirement)}')
m = gp.Model('MinWaitstaff')
shift_start_indices = list(range(48))
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in range(48):
    covering_starts = [s for s in range(48) if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((x_vars[s].X for s in shift_start_indices))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f"  Start at '{period_idx_to_label[s]}': {num} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')