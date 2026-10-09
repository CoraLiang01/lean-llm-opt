import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_periods = df.shape[0]
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
period_indices = list(range(num_periods))
shift_start_indices = list(range(num_periods))
if 'Requirement' not in df.columns:
    raise KeyError("Missing 'Requirement' column in CSV")
requirements = df['Requirement'].astype(int).to_dict()
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in period_indices:
    covering_shifts = []
    for s in shift_start_indices:
        if (t - s) % num_periods < 16:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shifts cover period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s].X for s in shift_start_indices))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in shift_start_indices:
        staff = int(round(x_vars[s].X))
        if staff > 0:
            time_label = df.iloc[s]['Time']
            print(f'  Start at period {s} ({time_label}): {staff} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')