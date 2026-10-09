import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {norm_col(c): c for c in df.columns}
shift_col = col_map.get('shift', None)
number_required_col = col_map.get('number required', None)
if shift_col is None or number_required_col is None:
    raise KeyError("Required columns 'Shift' and/or 'Number Required' not found in CSV.")
df['Shift_int'] = df[shift_col].str.strip().astype(int)
df['Number_Required_int'] = df[number_required_col].str.strip().astype(int)
all_hours = sorted(df['Shift_int'].unique())
if set(all_hours) != set(range(1, 25)):
    raise ValueError(f'Expected hours 1..24, got {all_hours}')
required_staff = dict(zip(df['Shift_int'], df['Number_Required_int']))
hours = list(range(1, 25))
m = gp.Model('BusRouteStaffing')
x_vars = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in hours)), gp.GRB.MINIMIZE)
for h in hours:
    covered_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covered_starts)) >= required_staff[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum staff assigned)')
    print('--- Staff Start Schedule (hour: staff assigned) ---')
    for t in hours:
        val = x_vars[t].X
        if val > 1e-06:
            print(f'  Hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')