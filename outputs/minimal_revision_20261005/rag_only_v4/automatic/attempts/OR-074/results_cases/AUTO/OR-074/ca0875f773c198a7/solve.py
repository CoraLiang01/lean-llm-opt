import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    req_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(req_path, sep=',')
    if set(df.columns) != {'Time', 'Requirement'}:
        raise ValueError(f'Unexpected columns in {req_path}: {df.columns}')
    periods = list(df.index)
    if len(periods) != 48:
        raise ValueError(f'Expected 48 periods, got {len(periods)}')
    requirement = df['Requirement'].astype(int)
    if requirement.isnull().any():
        raise ValueError('Missing requirement values in CSV')
    shift_starts = periods
    shift_length = 16
    coverage = {t: [] for t in periods}
    for s in shift_starts:
        covered = [(s + i) % 48 for i in range(shift_length)]
        for t in covered:
            coverage[t].append(s)
    m = gp.Model('min_waitstaff')
    m.Params.MIPGap = 0.0001
    x = m.addVars(shift_starts, vtype=GRB.INTEGER, lb=0, obj=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in coverage[t])) >= int(requirement.iloc[t]), name=f'cover_{t}')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for s in shift_starts:
            var = x[s]
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()