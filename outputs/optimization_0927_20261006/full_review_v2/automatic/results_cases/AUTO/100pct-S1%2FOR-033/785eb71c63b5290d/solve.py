import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
time_periods = list(df['Time'])
n_periods = len(time_periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 time periods, got {n_periods}')
period_idx_to_label = {i: time_periods[i] for i in range(n_periods)}
label_to_period_idx = {label: i for (i, label) in period_idx_to_label.items()}
try:
    requirement = df['Requirement'].astype(int).tolist()
except Exception as e:
    raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
if len(requirement) != n_periods:
    raise ValueError('Requirement length does not match number of periods.')
shift_start_indices = list(range(n_periods))
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
shift_length = 16
for t in range(n_periods):
    covering_s = []
    for s in shift_start_indices:
        covered = [(s + offset) % n_periods for offset in range(shift_length)]
        if t in covered:
            covering_s.append(s)
    if not covering_s:
        raise ValueError(f'No shift covers period {t} ({period_idx_to_label[t]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_s)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Assignment ---')
    for s in shift_start_indices:
        n_staff = int(round(x_vars[s].X))
        if n_staff > 0:
            shift_start_label = period_idx_to_label[s]
            shift_end_idx = (s + shift_length - 1) % n_periods
            shift_end_label = period_idx_to_label[shift_end_idx]
            print(f'Start: {shift_start_label:>15} | Staff: {n_staff:2d} | Covers: {shift_start_label} to {shift_end_label}')
else:
    print(f'No optimal solution found. Status: {m.status}')