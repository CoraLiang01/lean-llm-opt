import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_slots = list(df.index)
shift_starts = list(range(48))
if 'Requirement' not in df.columns:
    raise KeyError("Column 'Requirement' not found in CSV.")
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
shift_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in time_slots:
    covering_shifts = [s for s in shift_starts if (t - s) % 48 in range(16)]
    m.addConstr(gp.quicksum((shift_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((shift_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((shift_vars[s].X for s in shift_starts))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in shift_starts:
        num = int(round(shift_vars[s].X))
        if num > 0:
            time_label = df.loc[s, 'Time'] if 'Time' in df.columns else f'Slot {s}'
            print(f'  Start at {time_label}: {num} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')