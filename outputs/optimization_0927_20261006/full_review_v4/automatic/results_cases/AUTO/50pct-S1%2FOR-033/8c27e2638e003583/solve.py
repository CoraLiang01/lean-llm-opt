import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
period_idx_to_time = {i: periods[i] for i in range(len(periods))}
time_to_period_idx = {periods[i]: i for i in range(len(periods))}
num_periods = len(periods)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
try:
    requirement_per_period = {i: int(df.loc[i, 'Requirement']) for i in range(num_periods)}
except Exception as e:
    raise ValueError(f"Failed to parse 'Requirement' column as int: {e}")
shift_starts = list(range(num_periods))
shift_coverage = dict()
for s in shift_starts:
    covered = [(s + k) % num_periods for k in range(16)]
    shift_coverage[s] = set(covered)
period_covered_by_shifts = dict()
for t in range(num_periods):
    period_covered_by_shifts[t] = set()
    for s in shift_starts:
        if t in shift_coverage[s]:
            period_covered_by_shifts[t].add(s)
    if not period_covered_by_shifts[t]:
        raise ValueError(f'Period {t} ({period_idx_to_time[t]}) is not covered by any shift.')
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in range(num_periods):
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by_shifts[t])) >= requirement_per_period[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_waitstaff = sum((int(round(x_vars[s].X)) for s in shift_starts))
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff needed)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_starts:
        num_staff = int(round(x_vars[s].X))
        if num_staff > 0:
            shift_start_time = period_idx_to_time[s]
            shift_end_idx = (s + 16) % num_periods
            shift_end_time = period_idx_to_time[shift_end_idx]
            print(f"  {num_staff} staff start at '{shift_start_time}' (covering 8 hours to '{shift_end_time}')")
else:
    print(f'No optimal solution found. Status: {m.status}')