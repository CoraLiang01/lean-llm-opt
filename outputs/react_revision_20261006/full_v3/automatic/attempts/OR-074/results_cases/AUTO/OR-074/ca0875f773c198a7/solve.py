import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise ValueError("CSV missing required columns 'Time' and/or 'Requirement'.")
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError('Expected 48 periods (rows) in the CSV, got {}'.format(len(periods)))
requirements = df['Requirement'].astype(int).to_dict()
num_periods = 48
shift_length = 16
m = gp.Model('WaitstaffScheduling')
x = m.addVars(periods, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in periods:
    covering_starts = [s for s in periods if (t - s) % num_periods < shift_length]
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
m.setObjective(gp.quicksum((x[s] for s in periods)), gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for s in periods:
        print(f'{x[s].VarName} {x[s].X}')
else:
    print(f'Solver status: {m.status}')