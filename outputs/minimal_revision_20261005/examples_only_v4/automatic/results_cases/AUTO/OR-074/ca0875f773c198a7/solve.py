import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
expected_cols = {'Time', 'Requirement'}
if set(df.columns) != expected_cols:
    raise ValueError(f'CSV columns {df.columns.tolist()} do not match expected {expected_cols}')
periods = list(df.index)
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_to_time = dict(zip(periods, df['Time']))
requirements = df['Requirement'].to_dict()
num_periods = 48
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((x[t] for t in periods)), gp.GRB.MINIMIZE)
for p in periods:
    covered_starts = [(p - i) % num_periods for i in range(shift_length)]
    m.addConstr(gp.quicksum((x[t] for t in covered_starts)) >= requirements[p], name=f'cov{p}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in periods:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')