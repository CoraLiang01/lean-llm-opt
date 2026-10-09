import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
shift_length = 16
shift_covers = dict()
for s in periods:
    covered = [(s + i) % 48 for i in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = dict()
for t in periods:
    period_covered_by[t] = set()
    for s in periods:
        if t in shift_covers[s]:
            period_covered_by[t].add(s)
    if not period_covered_by[t]:
        raise ValueError(f'Period {t} is not covered by any shift start.')
m = gp.Model('MinWaitstaffShifts')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in periods))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        num = int(round(x_vars[s].X))
        if num > 0:
            time_label = df.iloc[s]['Time']
            print(f'Shift start at period {s} ({time_label}): {num} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')