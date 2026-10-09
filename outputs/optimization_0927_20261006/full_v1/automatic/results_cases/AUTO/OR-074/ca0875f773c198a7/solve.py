import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {label: i for (i, label) in period_idx_to_label.items()}
try:
    requirements = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
if len(requirements) != 48:
    raise ValueError(f'Expected 48 requirements, got {len(requirements)}')
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
for t in range(48):
    covering_starts = []
    for s in range(48):
        covered_periods = [(s + offset) % 48 for offset in range(shift_length)]
        if t in covered_periods:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_label[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(48):
        n = x_vars[s].X
        if n > 1e-06:
            print(f'  Start at {period_idx_to_label[s]}: {int(round(n))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')