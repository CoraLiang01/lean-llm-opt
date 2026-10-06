import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise ValueError("CSV file must contain 'Time' and 'Requirement' columns.")
periods = list(df.index)
if len(periods) != 48:
    raise ValueError('Expected 48 periods (rows) in the CSV file.')
period_to_time = dict(zip(df.index, df['Time']))
period_to_req = dict(zip(df.index, df['Requirement']))
shift_starts = list(df.index)
shift_length = 16
cover = np.zeros((48, 48), dtype=int)
for s in shift_starts:
    for i in range(shift_length):
        t = (s + i) % 48
        cover[t, s] = 1

def solve_staffing_minimum():
    m = gp.Model('MinWaitstaff')
    m.Params.MIPGap = 0.0001
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        m.addConstr(gp.quicksum((x[s] for s in shift_starts if cover[t, s])) >= period_to_req[t], name=f'cov{t}')
    m.optimize()
    return m
m = solve_staffing_minimum()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')