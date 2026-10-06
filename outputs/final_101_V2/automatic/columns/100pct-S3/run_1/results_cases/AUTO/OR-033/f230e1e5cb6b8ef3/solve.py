import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
period_labels = list(df['Time'])
n_periods = len(period_labels)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_idx_to_label = {i: period_labels[i] for i in range(n_periods)}
period_label_to_idx = {label: i for i, label in period_idx_to_label.items()}
requirements = df['Requirement'].astype(int).to_dict()
req = {i: int(df.loc[i, 'Requirement']) for i in range(n_periods)}
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(n_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in range(n_periods):
    covering_starts = []
    for s in range(n_periods):
        covered = [(s + offset) % n_periods for offset in range(shift_length)]
        if t in covered:
            covering_starts.append(s)
    if not covering_starts:
        raise ValueError(f'No shift starts cover period {t} ({period_idx_to_label[t]})')
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= req[t], name=f'cov_{t}')
m.setObjective(gp.quicksum((x[s] for s in range(n_periods))), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((x[s].X for s in range(n_periods)))
    print(f'Optimal total value/cost: {total_staff:.0f} (minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in range(n_periods):
        num = int(round(x[s].X))
        if num > 0:
            print(f'  Start at {period_idx_to_label[s]}: {num} staff')
    print('\n--- Coverage Check (period, required, scheduled) ---')
    for t in range(n_periods):
        staff_on_duty = sum((int(round(x[s].X)) for s in range(n_periods) if t in [(s + offset) % n_periods for offset in range(shift_length)]))
        print(f'  {period_idx_to_label[t]}: required={req[t]}, scheduled={staff_on_duty}')
else:
    print(f'No optimal solution found. Status: {m.status}')