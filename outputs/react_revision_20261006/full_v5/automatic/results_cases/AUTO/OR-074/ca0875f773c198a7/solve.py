import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise ValueError("CSV file must contain 'Time' and 'Requirement' columns.")
time_periods = list(df['Time'])
requirements = dict(zip(df['Time'], df['Requirement']))
if len(time_periods) != 48:
    raise ValueError(f'Expected 48 time periods, got {len(time_periods)}.')
shift_length = 16
num_periods = len(time_periods)
period_idx = {t: i for (i, t) in enumerate(time_periods)}
m = gp.Model('WaitstaffScheduling')
x = m.addVars(time_periods, vtype=gp.GRB.INTEGER, lb=0, name='')
for t in time_periods:
    t_idx = period_idx[t]
    covering_starts = []
    for offset in range(shift_length):
        s_idx = (t_idx - offset) % num_periods
        s = time_periods[s_idx]
        covering_starts.append(s)
    m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t_idx}')
m.setObjective(gp.quicksum((x[t] for t in time_periods)), gp.GRB.MINIMIZE)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for t in time_periods:
        print(f'{x[t].VarName} {x[t].X}')
else:
    print(f'Solver status: {m.status}')