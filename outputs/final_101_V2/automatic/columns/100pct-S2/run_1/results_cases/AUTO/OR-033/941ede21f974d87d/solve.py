import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(df['Time'])
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_idx_to_time = {i: periods[i] for i in range(n_periods)}
time_to_period_idx = {periods[i]: i for i in range(n_periods)}
requirements = df['Requirement'].astype(int).tolist()
if len(requirements) != n_periods:
    raise ValueError('Mismatch between periods and requirements length.')
shift_starts = list(range(n_periods))
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
shift_length = 16
for t in range(n_periods):
    covering_shifts = []
    for s in shift_starts:
        covered = [(s + offset) % n_periods for offset in range(shift_length)]
        if t in covered:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_time[t]})')
    m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in shift_starts))
    print(f'Optimal total value/cost: {total_staff:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Assignments ---')
    for s in shift_starts:
        staff_count = int(round(x[s].X))
        if staff_count > 0:
            print(f"Shift starting at '{period_idx_to_time[s]}': {staff_count} waitstaff")
    print('\n--- Coverage Check (period, requirement, covered) ---')
    for t in range(n_periods):
        covering_shifts = [(s, int(round(x[s].X))) for s in shift_starts if (t - s) % n_periods < shift_length]
        covered = sum((cnt for s, cnt in covering_shifts))
        print(f'{period_idx_to_time[t]}: required={requirements[t]}, covered={covered}')
else:
    print(f'No optimal solution found. Status: {m.status}')