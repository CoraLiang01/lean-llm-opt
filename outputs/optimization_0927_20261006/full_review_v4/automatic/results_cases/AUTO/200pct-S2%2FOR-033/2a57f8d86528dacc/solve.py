import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_periods = df.shape[0]
period_indices = list(range(num_periods))
if 'Requirement' not in df.columns:
    raise KeyError("Required column 'Requirement' not found in CSV.")
requirements = df['Requirement'].astype(int).to_dict()
shift_indices = period_indices
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_indices)), gp.GRB.MINIMIZE)
shift_length = 16
for t in period_indices:
    covering_shifts = []
    for s in shift_indices:
        covered = [(s + offset) % num_periods for offset in range(shift_length)]
        if t in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in shift_indices:
        val = x_vars[s].X
        if val > 0.5:
            time_label = df.loc[s, 'Time'] if 'Time' in df.columns else f'Period {s}'
            print(f'  Start at {time_label}: {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')