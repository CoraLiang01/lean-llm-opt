import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv', dtype=str, keep_default_na=False)
    required_cols = {'Shift', 'Time', 'Number Required'}
    if not required_cols.issubset(df.columns):
        raise KeyError(f'Missing required columns: {required_cols - set(df.columns)}')
    df['Shift'] = df['Shift'].str.strip()
    df['Number Required'] = df['Number Required'].str.strip()
    try:
        df['Shift'] = df['Shift'].astype(int)
        df['Number Required'] = df['Number Required'].astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert 'Shift' or 'Number Required' to int: {e}")
    shifts = sorted(df['Shift'].unique())
    if set(shifts) != set(range(1, 25)):
        raise ValueError(f'Expected shifts 1..24, got: {shifts}')
    required_staff = dict(zip(df['Shift'], df['Number Required']))
    periods = list(range(1, 25))
    m = gp.Model('BusCrewScheduling')
    staff_start_vars = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
    for s in periods:
        covering_starts = [(s - k - 1) % 24 + 1 for k in range(4)]
        m.addConstr(gp.quicksum((staff_start_vars[t] for t in covering_starts)) >= required_staff[s], name=f'cover_{s}')
    m.setObjective(gp.quicksum((staff_start_vars[t] for t in periods)), gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')