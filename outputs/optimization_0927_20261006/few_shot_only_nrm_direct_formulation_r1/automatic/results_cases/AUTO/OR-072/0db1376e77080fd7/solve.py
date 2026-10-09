import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Shift', 'Time', 'Number Required']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
if len(df) != 24:
    raise ValueError(f'Expected 24 rows (one per hour), got {len(df)}.')
try:
    shift_indices = df['Shift'].astype(str).str.strip()
    if all((re.fullmatch('\\d+', s) for s in shift_indices)):
        hours = shift_indices.astype(int).tolist()
        if sorted(hours) != list(range(1, 25)):
            hours = list(range(1, 25))
    else:
        hours = list(range(1, 25))
except Exception:
    hours = list(range(1, 25))
number_required = {}
for (i, row) in enumerate(df.itertuples(index=False), 1):
    try:
        req = int(str(row[df.columns.get_loc('Number Required')]).strip())
    except Exception as e:
        raise ValueError(f"Invalid 'Number Required' value at row {i}: {e}")
    number_required[i] = req
m = gp.Model('BusStaffScheduling')
x_vars = m.addVars(range(1, 25), vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[t] for t in range(1, 25))), gp.GRB.MINIMIZE)
for h in range(1, 25):
    covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
    m.addConstr(gp.quicksum((x_vars[t] for t in covering_starts)) >= number_required[h], name=f'cover_{h}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (minimum staff assigned)')
    print('--- Staff Assignment Plan ---')
    for t in range(1, 25):
        val = x_vars[t].X
        if val > 1e-06:
            print(f'  Start at hour {t}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')