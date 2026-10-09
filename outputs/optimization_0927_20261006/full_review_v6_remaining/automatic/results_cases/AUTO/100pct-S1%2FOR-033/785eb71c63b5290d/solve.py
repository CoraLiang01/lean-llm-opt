import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_indices = list(range(len(df)))
num_periods = len(period_indices)
period_labels = df['Time'].tolist()
if 'Requirement' not in df.columns:
    raise KeyError("Missing 'Requirement' column in input CSV.")
requirement_per_period = df['Requirement'].astype(int).to_dict()
shift_length = 16
shift_covers = dict()
for s in period_indices:
    covered = [(s + offset) % num_periods for offset in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in period_indices}
for s in period_indices:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(period_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in period_indices)), gp.GRB.MINIMIZE)
for t in period_indices:
    covering_shifts = period_covered_by[t]
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t} ({period_labels[t]}).')
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in covering_shifts)) >= requirement_per_period[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in period_indices:
        val = shift_start_vars[s].X
        if val > 1e-06:
            print(f'  Start at period {s} ({period_labels[s]}): {int(round(val))} waitstaff')
else:
    print(f'No optimal solution found. Status: {m.status}')