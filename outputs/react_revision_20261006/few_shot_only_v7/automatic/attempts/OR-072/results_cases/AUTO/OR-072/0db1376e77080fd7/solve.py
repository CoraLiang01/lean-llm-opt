import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_cols = {'Shift', 'Time', 'Number Required'}
if not required_cols.issubset(df.columns):
    missing = required_cols - set(df.columns)
    raise ValueError(f'Missing required columns in CSV: {missing}')
try:
    df['Time_int'] = df['Time'].astype(int)
except Exception:
    raise ValueError("Column 'Time' must contain integer hour indices (1-24).")
if not set(df['Time_int']) == set(range(1, 25)):
    raise ValueError('CSV must contain exactly one row for each hour 1-24.')
number_required = {}
for (idx, row) in df.iterrows():
    t = int(row['Time_int'])
    try:
        req = int(row['Number Required'])
    except Exception:
        raise ValueError(f"Non-integer value in 'Number Required' at row {idx + 1}.")
    number_required[t] = req
time_periods = list(range(1, 25))

def solve_bus_staffing(number_required, time_periods):
    m = gp.Model('BusStaffing')
    staff_vars = m.addVars(time_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((staff_vars[t] for t in time_periods)), gp.GRB.MINIMIZE)
    for t in time_periods:
        covered = [(t - i - 1) % 24 + 1 for i in range(4)]
        m.addConstr(gp.quicksum((staff_vars[tt] for tt in covered)) >= number_required[t], name=f'cov{t}')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_bus_staffing(number_required, time_periods)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')