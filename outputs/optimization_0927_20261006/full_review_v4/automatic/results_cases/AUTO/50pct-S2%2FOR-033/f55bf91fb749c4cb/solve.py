import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df['Time'])
if len(period_ids) != 48:
    raise ValueError(f'Expected 48 periods, got {len(period_ids)}')
period_idx_to_id = {i: period_ids[i] for i in range(48)}
period_id_to_idx = {period_ids[i]: i for i in range(48)}
try:
    requirement_per_period = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
except Exception as e:
    raise ValueError(f"Failed to parse 'Requirement' column as int: {e}")
m = gp.Model('MinWaitstaff')
shift_start_indices = list(range(48))
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
shift_length = 16
for t in range(48):
    covering_shift_starts = [s for s in range(48) if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shift_starts)) >= requirement_per_period[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        n = int(round(x_vars[s].X))
        if n > 0:
            print(f'  {period_idx_to_id[s]}: {n} waitstaff start shift')
    print('\n--- Coverage per period (requirement vs. actual) ---')
    for t in range(48):
        actual = sum((int(round(x_vars[s].X)) for s in range(48) if (t - s) % 48 < shift_length))
        print(f'  {period_idx_to_id[t]}: required {requirement_per_period[t]}, scheduled {actual}')
else:
    print(f'No optimal solution found. Status: {m.status}')