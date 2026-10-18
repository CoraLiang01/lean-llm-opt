import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
period_col = None
requirement_col = None
for col in df.columns:
    if re.fullmatch('Time', col.strip(), re.IGNORECASE) or re.search('time', col.strip(), re.IGNORECASE):
        period_col = col
    if re.fullmatch('Requirement', col.strip(), re.IGNORECASE) or re.search('requirement', col.strip(), re.IGNORECASE):
        requirement_col = col
if period_col is None:
    period_col = df.columns[0]
if requirement_col is None:
    requirement_col = df.columns[1]
periods = df[period_col].apply(lambda x: x.strip()).tolist()
requirements = df[requirement_col].apply(lambda x: int(x.strip())).tolist()
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_idx_to_id = {i: periods[i] for i in range(len(periods))}
period_id_to_idx = {periods[i]: i for i in range(len(periods))}
period_idx_to_req = {i: requirements[i] for i in range(len(requirements))}
shift_start_indices = list(range(len(periods)))
shift_coverage = dict()
num_periods_per_shift = 16
for s in shift_start_indices:
    covered = [(s + offset) % 48 for offset in range(num_periods_per_shift)]
    shift_coverage[s] = set(covered)
period_covered_by_shifts = dict()
for t in range(48):
    period_covered_by_shifts[t] = set()
    for s in shift_start_indices:
        if t in shift_coverage[s]:
            period_covered_by_shifts[t].add(s)
    if not period_covered_by_shifts[t]:
        raise ValueError(f'Period {t} ({period_idx_to_id[t]}) is not covered by any shift.')
m = gp.Model('WaitstaffScheduling')
shift_start_vars = m.addVars(shift_start_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s] for s in shift_start_indices)), gp.GRB.MINIMIZE)
for t in range(48):
    m.addConstr(gp.quicksum((shift_start_vars[s] for s in period_covered_by_shifts[t])) >= period_idx_to_req[t], name=f'cover_{t}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_staff = sum((shift_start_vars[s].X for s in shift_start_indices))
    print(f'Optimal total value/cost: {total_staff:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s in shift_start_indices:
        num_staff = int(round(shift_start_vars[s].X))
        if num_staff > 0:
            print(f"  Shift starting at '{period_idx_to_id[s]}': {num_staff} staff")
else:
    print(f'No optimal solution found. Status: {m.status}')