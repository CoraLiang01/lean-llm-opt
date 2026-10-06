import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if df.shape[0] != 48:
    raise ValueError(f'Expected 48 periods (rows), got {df.shape[0]}')
periods = list(df.index)
shift_starts = list(df.index)
if 'Requirement' not in df.columns:
    raise KeyError("Missing 'Requirement' column in CSV")
requirements = df['Requirement'].to_dict()
shift_length = 16
num_periods = len(periods)
shift_covers = {}
for s in shift_starts:
    covered = [(s + i) % num_periods for i in range(shift_length)]
    shift_covers[s] = set(covered)
period_covered_by = {t: set() for t in periods}
for s in shift_starts:
    for t in shift_covers[s]:
        period_covered_by[t].add(s)
for t in periods:
    if len(period_covered_by[t]) == 0:
        raise ValueError(f'Period {t} is not covered by any shift start.')

def solve_shift_scheduling():
    m = gp.Model('WaitstaffShiftScheduling')
    m.Params.MIPGap = 0.0001
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirements[t], name=f'cov_{t}')
    m.optimize()
    return m
m = solve_shift_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')