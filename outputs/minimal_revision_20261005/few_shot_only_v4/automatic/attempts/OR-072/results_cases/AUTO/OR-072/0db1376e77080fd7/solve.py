import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others1/42.csv'
df = pd.read_csv(csv_path, sep=',')
if 'Shift' in df.columns:
    time_col = 'Shift'
elif 'Time' in df.columns:
    time_col = 'Time'
else:
    raise KeyError("Neither 'Shift' nor 'Time' column found in CSV.")

def parse_hour(val):
    try:
        h = int(str(val).strip())
        if 1 <= h <= 24:
            return h
        else:
            raise ValueError
    except Exception:
        raise ValueError(f'Invalid hour label: {val}')
df['Hour'] = df[time_col].apply(parse_hour)
if set(df['Hour']) != set(range(1, 25)):
    missing = set(range(1, 25)) - set(df['Hour'])
    extra = set(df['Hour']) - set(range(1, 25))
    raise ValueError(f'CSV must contain exactly one row for each hour 1..24. Missing: {missing}, Extra: {extra}')
required_staff = df.set_index('Hour')['Number Required'].astype(int).to_dict()
hours = list(range(1, 25))

def solve_problem():
    m = gp.Model('BusRouteStaffing')
    x = m.addVars(hours, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in hours)), gp.GRB.MINIMIZE)
    for h in hours:
        covering_starts = [(h - k - 1) % 24 + 1 for k in range(4)]
        m.addConstr(gp.quicksum((x[t] for t in covering_starts)) >= required_staff[h], name=f'cov{h}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')