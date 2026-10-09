import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_labels = df['Time'].tolist()
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
shift_starts = periods
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in periods:
    covering_shifts = []
    for s in shift_starts:
        covered = [(s + i) % 48 for i in range(shift_length)]
        if t in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        num = int(round(x_vars[s].X))
        if num > 0:
            print(f'  Start at period {s:2d} ({period_labels[s]}): {num} waitstaff')
    print('\n--- Coverage by Period ---')
    for t in periods:
        covering = 0
        for s in shift_starts:
            covered = [(s + i) % 48 for i in range(shift_length)]
            if t in covered:
                covering += int(round(x_vars[s].X))
        print(f'  Period {t:2d} ({period_labels[t]}): Required={requirement[t]}, Covered={covering}')
else:
    print(f'No optimal solution found. Status: {m.status}')