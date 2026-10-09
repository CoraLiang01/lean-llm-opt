import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
hour_col = None
for col in df.columns:
    if re.fullmatch('(Shift|Time)', col, re.IGNORECASE):
        hour_col = col
        break
if hour_col is None:
    raise KeyError("Could not find a column named 'Shift' or 'Time' in 42.csv.")
df['hour'] = df[hour_col].astype(int)
hours = sorted(df['hour'].unique())
if set(hours) != set(range(1, 25)):
    raise ValueError('Expected hours 1-24 in 42.csv, got: %s' % sorted(hours))
if 'Number Required' not in df.columns:
    raise KeyError("Could not find 'Number Required' column in 42.csv.")
required_staff = df.set_index('hour')['Number Required'].astype(int).to_dict()
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in hours)), gp.GRB.MINIMIZE)
for h in hours:
    covered_by = [(h - i - 1) % 24 + 1 for i in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covered_by)) >= required_staff[h], name=f'cover_{h}')
m.optimize()