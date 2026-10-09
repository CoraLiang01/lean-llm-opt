import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_periods = df.shape[0]
if num_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {num_periods}')
period_indices = list(range(num_periods))
if 'Requirement' not in df.columns:
    raise KeyError("Column 'Requirement' not found in CSV")
requirements = df['Requirement'].astype(int).to_dict()
shift_length = 16
shift_covers = dict()
for s in period_indices:
    covered = [(s + offset) % num_periods for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = dict()
for t in period_indices:
    period_covered_by[t] = [s for s in period_indices if t in shift_covers[s]]
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(period_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in period_indices)), gp.GRB.MINIMIZE)
for t in period_indices:
    covering_shifts = period_covered_by[t]
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((shift_start_vars[s].X for s in period_indices))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule ---')
    for s in period_indices:
        num_staff = int(round(shift_start_vars[s].X))
        if num_staff > 0:
            time_label = df.loc[s, 'Time'] if 'Time' in df.columns else f'Period {s}'
            print(f'  Start at {time_label}: {num_staff} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')