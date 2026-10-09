import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
expected_cols = {'Time', 'Requirement'}
if set(df.columns) != expected_cols:
    raise ValueError(f'CSV columns {df.columns.tolist()} do not match expected {sorted(expected_cols)}')
periods = list(range(48))
if len(df) != 48:
    raise ValueError(f'Expected 48 periods (rows), got {len(df)}')
try:
    requirement = {}
    for (idx, row) in df.iterrows():
        period = idx
        req = int(row['Requirement'])
        requirement[period] = req
except Exception as e:
    raise ValueError(f"Error parsing 'Requirement' column: {e}")
shift_starts = periods

def solve_staff_scheduling(periods, shift_starts, requirement):
    m = gp.Model('WaitstaffScheduling')
    x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    shift_length = 16
    for t in periods:
        covering_starts = [(t - offset) % 48 for offset in range(shift_length)]
        m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirement[t], name=f'cover_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_staff_scheduling(periods, shift_starts, requirement)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')