import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
requirement = {}
for (idx, row) in df.iterrows():
    period = row['Time']
    try:
        req = int(row['Requirement'])
    except Exception as e:
        raise ValueError(f"Invalid Requirement value at period '{period}': {row['Requirement']}") from e
    requirement[period] = req
shift_starts = periods
period_to_idx = {period: idx for (idx, period) in enumerate(periods)}
idx_to_period = {idx: period for (idx, period) in enumerate(periods)}
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
for (t_idx, t) in enumerate(periods):
    covering_shift_starts = []
    for (s_idx, s) in enumerate(shift_starts):
        covered_indices = [(s_idx + offset) % 48 for offset in range(16)]
        if t_idx in covered_indices:
            covering_shift_starts.append(s)
    if not covering_shift_starts:
        raise ValueError(f"No shift covers period '{t}' (index {t_idx})")
    m.addConstr(gp.quicksum((x_vars[s] for s in covering_shift_starts)) >= requirement[t], name=f'cover_{t_idx}')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
m.optimize()