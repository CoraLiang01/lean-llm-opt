import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df.index)
num_periods = len(period_ids)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
requirement = {}
for idx in period_ids:
    req_val = df.loc[idx, 'Requirement']
    try:
        requirement[idx] = int(req_val)
    except Exception:
        raise ValueError(f'Invalid Requirement value at row {idx}: {req_val}')
shift_start_ids = period_ids.copy()
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_ids)), gp.GRB.MINIMIZE)
for t in period_ids:
    covering_shifts = []
    for s in shift_start_ids:
        covered = [(int(s) + offset) % num_periods for offset in range(shift_length)]
        if int(t) in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in shift_start_ids:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.loc[int(s), 'Time']
            print(f'  Shift starting at period {s} ({time_label}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')