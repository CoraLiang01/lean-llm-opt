import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(range(48))
shift_starts = list(range(48))
period_idx_to_time = dict(zip(periods, df['Time']))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods, got {len(df)} rows in the CSV.')
requirements = df['Requirement'].astype(int).tolist()
if any((r < 0 for r in requirements)):
    raise ValueError('Negative staffing requirement found.')
shift_length = 16
period_covered_by_shiftstart = {t: [] for t in periods}
for s in shift_starts:
    for offset in range(shift_length):
        t = (s + offset) % 48
        period_covered_by_shiftstart[t].append(s)
m = gp.Model('WaitstaffScheduling')
x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
for t in periods:
    covering_starts = period_covered_by_shiftstart[t]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in shift_starts:
        print(f'x[{s}] {x[s].VarName} {x[s].X}')
else:
    print(f'Solver status: {m.status}')