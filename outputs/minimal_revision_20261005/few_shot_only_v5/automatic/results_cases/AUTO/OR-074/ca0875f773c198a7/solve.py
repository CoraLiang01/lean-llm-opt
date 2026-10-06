import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
expected_cols = {'Time', 'Requirement'}
if set(df.columns) != expected_cols:
    raise ValueError(f'CSV columns {set(df.columns)} do not match expected {expected_cols}')
if not np.issubdtype(df['Time'].dtype, np.integer):
    df['Time'] = df['Time'].astype(int)
if not np.issubdtype(df['Requirement'].dtype, np.integer):
    df['Requirement'] = df['Requirement'].astype(int)
periods = sorted(df['Time'].unique())
if periods != list(range(24)):
    raise ValueError(f'Time periods in CSV are {periods}, expected 0..23 for 24-hour operation.')
requirements = dict(zip(df['Time'], df['Requirement']))
shift_starts = periods

def solve_problem(periods, requirements, shift_starts):
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        covering_starts = []
        for s in shift_starts:
            if 0 <= (t - s) % 24 <= 7:
                covering_starts.append(s)
        if not covering_starts:
            raise ValueError(f'No shift starts cover period {t}')
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(periods, requirements, shift_starts)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')