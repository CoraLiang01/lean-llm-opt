import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 time periods, got {len(periods)}.')
period_idx_to_time = {i: periods[i] for i in range(48)}
time_to_period_idx = {periods[i]: i for i in range(48)}
try:
    requirement = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
if len(requirement) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirement)}.')
shift_length = 16
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
for t in range(48):
    covering_shifts = [s for s in range(48) if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in range(48)))
    print(f'Optimal total value/cost: {total_waitstaff} (Minimum number of waitstaff)')
    print('\n--- Shift Assignments ---')
    for s in range(48):
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f'  Shift start: {period_idx_to_time[s]:>15} | Waitstaff: {num}')
else:
    print(f'No optimal solution found. Status: {m.status}')