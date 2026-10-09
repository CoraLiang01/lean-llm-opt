import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = df.index.tolist()
n_periods = len(period_ids)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_length = 16
shift_covers = dict()
for s in period_ids:
    covered = [(s + offset) % n_periods for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in period_ids}
for s in period_ids:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(period_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in period_ids)), gp.GRB.MINIMIZE)
for t in period_ids:
    covering_shifts = period_covered_by[t]
    if not covering_shifts:
        raise ValueError(f'Period {t} is not covered by any shift (should not happen)')
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((shift_start_vars[s].X for s in period_ids))
    print(f'Optimal total value/cost: {total_waitstaff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in period_ids:
        n = int(round(shift_start_vars[s].X))
        if n > 0:
            time_label = df.loc[s, 'Time']
            print(f'Start at period {s} ({time_label}): {n} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')