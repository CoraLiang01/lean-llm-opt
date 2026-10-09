import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {label: i for (i, label) in period_idx_to_label.items()}
try:
    requirement = [int(x) for x in df['Requirement']]
except Exception as e:
    raise ValueError(f'Could not convert all Requirement values to int: {e}')
if len(requirement) != 48:
    raise ValueError(f'Expected 48 requirement values, got {len(requirement)}')
shift_starts = list(range(48))
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in range(48):
    covering_shifts = []
    for s in shift_starts:
        covered = [(s + offset) % 48 for offset in range(16)]
        if t in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shifts cover period {t} ({period_idx_to_label[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Assignments ---')
    for s in shift_starts:
        val = x_vars[s].X
        if val > 1e-06:
            print(f'  Shift starting at period {s} ({period_idx_to_label[s]}): {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')