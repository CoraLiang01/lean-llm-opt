import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 periods, got {df.shape[0]} rows in {csv_path}')
periods = list(df['Time'])
if len(set(periods)) != 48:
    raise ValueError('Time periods are not unique in the input file.')
try:
    requirements = df.set_index('Time')['Requirement'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f'Could not extract integer requirements: {e}')
shift_starts = periods.copy()
period_idx = {p: i for (i, p) in enumerate(periods)}
idx_period = {i: p for (i, p) in enumerate(periods)}
num_periods = 48
shift_length = 16
shift_covers = dict()
for (s_idx, s) in enumerate(shift_starts):
    covered = set()
    for offset in range(shift_length):
        t_idx = (s_idx + offset) % num_periods
        covered.add(idx_period[t_idx])
    shift_covers[s] = covered
period_covered_by = dict()
for t in periods:
    t_idx = period_idx[t]
    covering_s = []
    for (s_idx, s) in enumerate(shift_starts):
        covered_idxs = [(s_idx + offset) % num_periods for offset in range(shift_length)]
        if t_idx in covered_idxs:
            covering_s.append(s)
    period_covered_by[t] = covering_s
for t in periods:
    if len(period_covered_by[t]) == 0:
        raise ValueError(f'Period {t} is not covered by any shift start.')
m = gp.Model('WaitstaffScheduling')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirements[t], name=f'cov_{period_idx[t]}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        var = x_vars[s]
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')