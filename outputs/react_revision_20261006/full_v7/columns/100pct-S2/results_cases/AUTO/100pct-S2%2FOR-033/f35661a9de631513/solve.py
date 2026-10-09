import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 time periods, got {len(periods)}')
if len(set(periods)) != 48:
    raise ValueError('Time period identifiers are not unique.')
try:
    requirement = {period: int(req) for (period, req) in zip(df['Time'], df['Requirement'])}
except Exception as e:
    raise ValueError(f"Error converting 'Requirement' to int: {e}")
num_periods = 48
shift_length = 16
shift_starts = list(range(num_periods))
period_idx = {period: idx for (idx, period) in enumerate(periods)}
idx_period = {idx: period for (idx, period) in enumerate(periods)}
shift_coverage = dict()
for s in shift_starts:
    covered_idxs = [(s + k) % num_periods for k in range(shift_length)]
    covered_periods = [idx_period[i] for i in covered_idxs]
    shift_coverage[s] = set(covered_periods)
period_covered_by = dict()
for (t_idx, t) in enumerate(periods):
    covering_shifts = []
    for s in shift_starts:
        if t in shift_coverage[s]:
            covering_shifts.append(s)
    if not covering_shifts:
        raise ValueError(f'Period {t} is not covered by any shift.')
    period_covered_by[t] = covering_shifts
m = gp.Model('MinWaitstaffShifts')
x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    m.addConstr(gp.quicksum((x_vars[s] for s in period_covered_by[t])) >= requirement[t], name=f'cov_{period_idx[t]}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        print(f'x[{s}] {x_vars[s].VarName} {x_vars[s].X}')
else:
    print(f'Solver status: {m.status}')