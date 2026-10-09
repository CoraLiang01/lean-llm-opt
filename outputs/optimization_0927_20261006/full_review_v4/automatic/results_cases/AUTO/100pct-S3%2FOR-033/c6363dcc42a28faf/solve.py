import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
period_ids = df['Time'].tolist()
period_idx_to_id = {i: period_ids[i] for i in range(len(period_ids))}
period_id_to_idx = {period_ids[i]: i for i in range(len(period_ids))}
num_periods = len(period_ids)
requirement = {}
for (i, row) in df.iterrows():
    period_id = row['Time']
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period_id}': {row['Requirement']}")
    requirement[period_id] = req
shift_start_ids = period_ids.copy()
shift_coverage = {}
for (s_idx, s_id) in enumerate(shift_start_ids):
    covered = []
    for offset in range(16):
        covered_idx = (s_idx + offset) % num_periods
        covered.append(period_idx_to_id[covered_idx])
    shift_coverage[s_id] = set(covered)
period_covered_by_shifts = {pid: [] for pid in period_ids}
for s_id in shift_start_ids:
    for pid in shift_coverage[s_id]:
        period_covered_by_shifts[pid].append(s_id)
m = gp.Model('MinWaitstaff')
x_vars = m.addVars(shift_start_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s_id] for s_id in shift_start_ids)), gp.GRB.MINIMIZE)
for pid in period_ids:
    covering_shifts = period_covered_by_shifts[pid]
    m.addConstr(gp.quicksum((x_vars[s_id] for s_id in covering_shifts)) >= requirement[pid], name=f'cover_{pid}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s_id in shift_start_ids:
        val = x_vars[s_id].X
        if val > 0.5:
            print(f'  Shift starting at {s_id}: {int(round(val))} staff')
else:
    print(f'No optimal solution found. Status: {m.status}')