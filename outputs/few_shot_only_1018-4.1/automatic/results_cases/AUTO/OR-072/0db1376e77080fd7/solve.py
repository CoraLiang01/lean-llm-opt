import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def find_col(df, pattern):
    for col in df.columns:
        if re.fullmatch(pattern, col.strip(), re.IGNORECASE):
            return col
    for col in df.columns:
        if re.search(pattern, col.strip(), re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
shift_col = find_col(df, 'Shift')
required_col = find_col(df, 'Number\\s*Required')
df[shift_col] = df[shift_col].str.strip()
try:
    hours = [int(s) for s in df[shift_col]]
except Exception as e:
    raise ValueError(f'Could not convert shift numbers to integers: {e}')
if len(hours) != 24 or sorted(hours) != list(range(1, 25)):
    raise ValueError('Expected 24 unique hourly shifts indexed 1..24.')
df[required_col] = df[required_col].str.strip()
try:
    required_staff = {int(row[shift_col]): int(row[required_col]) for _, row in df.iterrows()}
except Exception as e:
    raise ValueError(f'Could not convert required staff to integers: {e}')
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in hours)), gp.GRB.MINIMIZE)
for h in hours:
    covering_ts = [t for t in hours if (h - t) % 24 in {0, 1, 2, 3}]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_ts)) >= required_staff[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum staff assigned)')
    print('--- Shift Assignment ---')
    for t in hours:
        val = x_vars[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')