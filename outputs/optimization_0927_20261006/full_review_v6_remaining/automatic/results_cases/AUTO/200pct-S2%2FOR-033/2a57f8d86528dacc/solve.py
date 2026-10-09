import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
num_periods = df.shape[0]
period_indices = list(range(num_periods))
shift_start_indices = list(range(num_periods))
requirement = df['Requirement'].astype(int).to_dict()
shift_covers = dict()
for s in shift_start_indices:
    covered = [(s + offset) % num_periods for offset in range(16)]
    shift_covers[s] = set(covered)
period_covered_by_shifts = dict()
for t in period_indices:
    period_covered_by_shifts[t] = set()
    for s in shift_start_indices:
        if t in shift_covers[s]:
            period_covered_by_shifts[t].add(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in period_indices:
    covering_shifts = period_covered_by_shifts[t]
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('--- Shift Start Schedule (nonzero only) ---')
    for s in shift_start_indices:
        val = x_vars[s].X
        if val > 1e-06:
            shift_time = df.iloc[s]['Time']
            print(f'Shift starting at period {s} ({shift_time}): {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')