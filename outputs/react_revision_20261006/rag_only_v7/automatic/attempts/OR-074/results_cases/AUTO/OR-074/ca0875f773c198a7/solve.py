import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Time' not in df.columns or 'Requirement' not in df.columns:
    raise ValueError("CSV must contain 'Time' and 'Requirement' columns.")
time_periods = df['Time'].tolist()
if len(time_periods) != 48:
    raise ValueError('Expected 48 time periods, got {}'.format(len(time_periods)))
if len(set(time_periods)) != 48:
    raise ValueError('Time periods must be unique.')
try:
    requirements = {row['Time']: int(row['Requirement']) for (_, row) in df.iterrows()}
except Exception as e:
    raise ValueError("Failed to convert 'Requirement' to int: {}".format(e))
shift_starts = list(time_periods)
num_periods = len(time_periods)
shift_length = 16
period_idx = {tp: idx for (idx, tp) in enumerate(time_periods)}
shift_covers = {}
for (s_idx, s) in enumerate(shift_starts):
    covered = []
    for offset in range(shift_length):
        t_idx = (s_idx + offset) % num_periods
        covered.append(time_periods[t_idx])
    shift_covers[s] = set(covered)
period_covered_by = {t: [] for t in time_periods}
for s in shift_starts:
    for t in shift_covers[s]:
        period_covered_by[t].append(s)
for (t, covering_shifts) in period_covered_by.items():
    if not covering_shifts:
        raise ValueError(f'Time period {t} is not covered by any shift.')

def solve_problem():
    m = gp.Model('waitstaff_scheduling')
    m.Params.MIPGap = 0.0001
    shift_vars = m.addVars(shift_starts, vtype=GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((shift_vars[s] for s in shift_starts)), GRB.MINIMIZE)
    for t in time_periods:
        m.addConstr(gp.quicksum((shift_vars[s] for s in period_covered_by[t])) >= requirements[t], name='cov_')
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.Status}')