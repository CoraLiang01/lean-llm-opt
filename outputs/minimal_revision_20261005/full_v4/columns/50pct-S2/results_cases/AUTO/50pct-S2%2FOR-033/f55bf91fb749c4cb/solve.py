import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(range(len(df)))
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods, got {len(periods)}')
period_to_time = dict(zip(periods, df['Time']))
requirement = df['Requirement'].astype(int).to_dict()
num_periods = 48
shift_length = 16
shift_starts = periods

def solve_waitstaff_scheduling():
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in periods:
        covering_starts = [(t - i) % num_periods for i in range(shift_length)]
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirement[t], name=f'cov_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_waitstaff_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')