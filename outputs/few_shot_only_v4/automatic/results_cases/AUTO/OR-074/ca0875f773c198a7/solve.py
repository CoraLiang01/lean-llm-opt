import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def find_col(df, pattern):
    for col in df.columns:
        if re.search(pattern, col, re.IGNORECASE):
            return col
    raise KeyError(f"Could not find a column matching pattern '{pattern}'")
period_col = find_col(df, 'period')
requirement_col = find_col(df, 'require')
periods = df[period_col].astype(str).str.strip().tolist()
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods (half-hour slots in 24h), got {len(periods)}')
period_idx_to_id = {i: periods[i] for i in range(len(periods))}
period_id_to_idx = {pid: i for i, pid in period_idx_to_id.items()}
requirements = {}
for idx, row in df.iterrows():
    pid = str(row[period_col]).strip()
    try:
        req = int(row[requirement_col])
    except Exception as e:
        raise ValueError(f"Requirement value for period '{pid}' is not a valid integer: {row[requirement_col]}")
    requirements[pid] = req
shift_starts = periods.copy()
coverage = {}
num_periods = len(periods)
shift_length = 16
for s_idx, s_id in enumerate(shift_starts):
    covered = []
    for offset in range(shift_length):
        t_idx = (s_idx + offset) % num_periods
        t_id = period_idx_to_id[t_idx]
        covered.append(t_id)
    coverage[s_id] = set(covered)
m = gp.Model('MinWaitstaff')
shift_start_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((shift_start_vars[s_id] for s_id in shift_starts)), gp.GRB.MINIMIZE)
for t_id in periods:
    covering_shifts = [s_id for s_id in shift_starts if t_id in coverage[s_id]]
    if not covering_shifts:
        raise ValueError(f'No shift covers period {t_id}')
    m.addConstr(gp.quicksum((shift_start_vars[s_id] for s_id in covering_shifts)) >= requirements[t_id], name=f'cover_{t_id}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Minimum number of waitstaff)')
    print('\n--- Shift Start Schedule ---')
    for s_id in shift_starts:
        val = shift_start_vars[s_id].X
        if val > 1e-06:
            print(f'  Shift starting at {s_id}: {int(round(val))} staff')
    print('\n--- Period Coverage Check ---')
    for t_id in periods:
        total = sum((shift_start_vars[s_id].X for s_id in shift_starts if t_id in coverage[s_id]))
        print(f'  Period {t_id}: Required {requirements[t_id]}, Covered {int(round(total))}')
else:
    print(f'No optimal solution found. Status: {m.status}')