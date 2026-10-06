import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_labels = dict(zip(periods, df['Time']))
requirements = df['Requirement'].to_dict()
if any((pd.isnull(requirements[p]) for p in periods)):
    raise ValueError('Missing requirement values in input data.')
shift_length = 16
num_periods = 48
shift_starts = periods

def solve_problem():
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        covering_shifts = []
        for s in shift_starts:
            if (t - s) % num_periods < shift_length:
                covering_shifts.append(s)
        if not covering_shifts:
            raise ValueError(f'No shift covers period {t} ({period_labels[t]})')
        m.addConstr(gp.quicksum((x[s] for s in covering_shifts)) >= requirements[t], name=f'cov_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')