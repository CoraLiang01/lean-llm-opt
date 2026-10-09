import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df.index)
num_periods = len(periods)
if num_periods != 48:
    raise ValueError(f'Expected 48 periods, got {num_periods}')
period_labels = df['Time'].tolist()
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_length = 16
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in periods)), gp.GRB.MINIMIZE)
for t in periods:
    covered_by = []
    t_int = int(t) if isinstance(t, (int, np.integer)) else int(t)
    for s in periods:
        s_int = int(s) if isinstance(s, (int, np.integer)) else int(s)
        covered_periods = [(s_int + offset) % num_periods for offset in range(shift_length)]
        if t_int in covered_periods:
            covered_by.append(s)
    if not covered_by:
        raise ValueError(f'No shift covers period {t} ({period_labels[t_int]})')
    m.addConstr(gp.quicksum((x_vars[s] for s in covered_by)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Assignments ---')
    for s in periods:
        num_staff = x_vars[s].X
        if num_staff > 1e-06:
            s_int = int(s) if isinstance(s, (int, np.integer)) else int(s)
            shift_start_label = period_labels[s_int]
            shift_end_int = (s_int + shift_length) % num_periods
            shift_end_label = period_labels[shift_end_int]
            print(f'  Start: {shift_start_label:>15} | End: {shift_end_label:>15} | Staff: {int(round(num_staff))}')
else:
    print(f'No optimal solution found. Status: {m.status}')