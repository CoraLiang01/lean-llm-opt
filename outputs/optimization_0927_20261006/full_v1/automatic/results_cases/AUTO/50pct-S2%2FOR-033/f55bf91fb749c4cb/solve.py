import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(range(48))
period_idx_to_time = dict(zip(periods, df['Time']))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods, found {len(df)} in the CSV.')
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
requirement_per_period = {i: int(df.iloc[i]['Requirement']) for i in periods}
shift_length = 16
shift_covers = dict()
for s in periods:
    covered = [(s + offset) % 48 for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in periods}
for s in periods:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirement_per_period[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x_vars[s].X for s in periods))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in periods:
        num_staff = int(round(x_vars[s].X))
        if num_staff > 0:
            time_label = period_idx_to_time[s]
            print(f"  Start at '{time_label}': {num_staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')