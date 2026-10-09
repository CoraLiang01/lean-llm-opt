import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = list(df.index)
n_periods = len(period_ids)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_labels = df['Time'].tolist()
if 'Requirement' not in df.columns:
    raise KeyError("Column 'Requirement' not found in CSV.")
try:
    requirement = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_start_ids = list(range(n_periods))
shift_coverage = {}
for s in shift_start_ids:
    covered = [(s + offset) % n_periods for offset in range(16)]
    shift_coverage[s] = set(covered)
period_covered_by = {t: set() for t in period_ids}
for s in shift_start_ids:
    for t in shift_coverage[s]:
        period_covered_by[t].add(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_ids)), gp.GRB.MINIMIZE)
for t in period_ids:
    covering_shifts = period_covered_by[t]
    if not covering_shifts:
        raise ValueError(f'Period {t} ({period_labels[t]}) is not covered by any shift.')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirement[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_ids:
        val = x_vars[s].X
        if val > 0.5:
            print(f'  Shift starting at period {s} ({period_labels[s]}): {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')