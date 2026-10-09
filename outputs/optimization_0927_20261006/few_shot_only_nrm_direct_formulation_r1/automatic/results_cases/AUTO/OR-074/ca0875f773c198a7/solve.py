import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in {csv_path}")
if len(df) != 48:
    raise ValueError(f'Expected 48 periods in {csv_path}, found {len(df)}')
period_ids = list(range(48))
try:
    requirements = df['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Failed to convert 'Requirement' column to int: {e}")
shift_start_ids = list(range(48))
shift_coverage = dict()
for s in shift_start_ids:
    covered = [(s + i) % 48 for i in range(16)]
    shift_coverage[s] = set(covered)
period_covered_by_shifts = {t: [] for t in period_ids}
for s in shift_start_ids:
    for t in shift_coverage[s]:
        period_covered_by_shifts[t].append(s)
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_start_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_start_ids)), gp.GRB.MINIMIZE)
for t in period_ids:
    covering_shifts = period_covered_by_shifts[t]
    if len(covering_shifts) == 0:
        raise ValueError(f'No shift covers period {t}')
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shifts)) >= requirements[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule (x_s) ---')
    for s in shift_start_ids:
        val = x_vars[s].X
        if val > 1e-06:
            time_label = df.loc[s, 'Time'] if 'Time' in df.columns else f'Period {s}'
            print(f'  Shift starting at {time_label}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')