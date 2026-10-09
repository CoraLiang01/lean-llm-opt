import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx = {label: idx for (idx, label) in enumerate(periods)}
try:
    requirement = {i: int(df.loc[i, 'Requirement']) for i in range(48)}
except Exception as e:
    raise ValueError(f"Error parsing 'Requirement' column: {e}")
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
shift_length = 16
for t in range(48):
    covering_starts = [s for s in range(48) if (t - s) % 48 < shift_length]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in range(48):
        n = x_vars[s].X
        if n > 1e-06:
            print(f"  Shift starting at '{periods[s]}': {int(round(n))} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')