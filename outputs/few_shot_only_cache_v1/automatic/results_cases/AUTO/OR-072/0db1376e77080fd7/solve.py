import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Shift' not in df.columns or 'Number Required' not in df.columns:
    raise KeyError("Required columns 'Shift' or 'Number Required' not found in CSV.")
periods = df['Shift'].astype(int).tolist()
if sorted(periods) != list(range(1, 25)):
    raise ValueError("Expected 24 consecutive periods indexed 1..24 in 'Shift' column.")
required_staff = {}
for idx, row in df.iterrows():
    shift = int(row['Shift'])
    try:
        req = int(row['Number Required'])
    except Exception as e:
        raise ValueError(f"Invalid 'Number Required' value at row {idx}: {row['Number Required']}")
    required_staff[shift] = req
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in periods)), gp.GRB.MINIMIZE)
for s in periods:
    covering_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= required_staff[s], name=f'cover_{s}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum total staff assigned)')
    print('--- Staff Shift Start Plan ---')
    for t in periods:
        val = x_vars[t].X
        if val > 1e-06:
            print(f"  Start at period {t} ({df.loc[df['Shift'].astype(int) == t, 'Time'].values[0]}): {int(round(val))} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')