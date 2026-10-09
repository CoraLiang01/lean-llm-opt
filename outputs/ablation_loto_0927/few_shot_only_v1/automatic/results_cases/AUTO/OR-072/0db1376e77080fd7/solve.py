import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Number Required' not in df.columns:
    raise KeyError("Column 'Number Required' not found in 42.csv")
if 'Shift' not in df.columns:
    raise KeyError("Column 'Shift' not found in 42.csv")
if len(df) != 24:
    raise ValueError(f'Expected 24 rows for 24 hours, got {len(df)} rows.')
periods = list(range(1, 25))
required_staff = {}
for (idx, row) in df.iterrows():
    t = idx + 1
    val = row['Number Required']
    try:
        required_staff[t] = int(val)
    except Exception:
        raise ValueError(f"Invalid value in 'Number Required' at row {idx}: {val}")
m = gp.Model('BusRouteStaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
for s in periods:
    covering_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required_staff[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum total staff assigned)')
    print('--- Staff Start Schedule (x[t]) ---')
    for t in periods:
        if x[t].X > 1e-06:
            print(f'  Start at period {t}: {int(round(x[t].X))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')