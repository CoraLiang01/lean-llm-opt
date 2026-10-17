import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def normalize_col(col):
    return col.strip().casefold()
col_map = {normalize_col(c): c for c in df.columns}
if 'hour' not in col_map or 'number required' not in col_map:
    raise KeyError("CSV must contain columns 'Hour' and 'Number Required' (case-insensitive).")
hour_col = col_map['hour']
req_col = col_map['number required']
df['__hour'] = df[hour_col].apply(lambda x: int(str(x).strip()))
df['__required'] = df[req_col].apply(lambda x: int(str(x).strip()))
hours = sorted(df['__hour'].unique())
if len(hours) != 24 or set(hours) != set(range(1, 25)):
    raise ValueError('CSV must contain exactly one row for each hour 1-24.')
required_staff = df.set_index('__hour')['__required'].to_dict()
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in hours)), gp.GRB.MINIMIZE)
for h in hours:
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= required_staff[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum staff assigned)')
    print('--- Shift Assignment ---')
    for t in hours:
        val = x_vars[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')