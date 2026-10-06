import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
periods = list(range(48))
shifts = list(range(48))
period_to_time = dict(zip(periods, df['Time']))
if df['Requirement'].isnull().any():
    raise ValueError("Missing values in 'Requirement' column.")
requirements = df['Requirement'].astype(int).to_dict()
if len(requirements) != 48:
    raise ValueError(f'Expected 48 periods, got {len(requirements)}.')
shift_covers = {s: [(s + k) % 48 for k in range(16)] for s in shifts}
period_covered_by = {t: [s for s in shifts if (t - s) % 48 in range(16)] for t in periods}
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shifts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shifts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in shifts))
    print(f'Optimal total value/cost: {total_staff:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shifts:
        staff_count = int(round(x[s].X))
        if staff_count > 0:
            time_label = period_to_time[s]
            covered_periods = [(s + k) % 48 for k in range(16)]
            covered_times = [period_to_time[p] for p in covered_periods]
            print(f'Shift start at period {s} ({time_label}): {staff_count} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')