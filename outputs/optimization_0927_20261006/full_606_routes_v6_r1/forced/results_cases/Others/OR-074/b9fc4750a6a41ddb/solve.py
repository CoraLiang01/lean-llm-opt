import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise KeyError("Required columns 'Time' and 'Requirement' not found in CSV.")
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}.')
period_idx_to_label = {i: periods[i] for i in range(48)}
period_label_to_idx = {label: i for (i, label) in period_idx_to_label.items()}
try:
    requirements = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
m = gp.Model('Waitstaff_Scheduling')
x_vars = m.addVars(range(48), vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in range(48):
    covering_starts = []
    for s in range(48):
        covered_periods = [(s + offset) % 48 for offset in range(shift_length)]
        if t in covered_periods:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_label[t]}).')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in range(48))), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(48):
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f"  Start at '{period_idx_to_label[s]}': {num} waitstaff")
else:
    print(f'No optimal solution found. Status: {m.status}')