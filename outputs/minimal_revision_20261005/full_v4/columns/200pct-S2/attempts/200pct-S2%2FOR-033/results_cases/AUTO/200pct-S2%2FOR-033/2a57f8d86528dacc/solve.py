import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(df.index)
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_to_time = df['Time'].to_dict()
requirement = df['Requirement'].astype(int).to_dict()
shift_length = 16
shift_starts = periods

def solve_shift_scheduling():
    m = gp.Model('WaitstaffScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[j] for j in shift_starts)), gp.GRB.MINIMIZE)
    for i in periods:
        covered_by = []
        for j in shift_starts:
            covered_periods = [(j + k) % n_periods for k in range(shift_length)]
            if i in covered_periods:
                covered_by.append(j)
        if not covered_by:
            raise ValueError(f'Period {i} ({period_to_time[i]}) is not covered by any shift start.')
        m.addConstr(gp.quicksum((x[j] for j in covered_by)) >= requirement[i], name=f'cov_{i}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_shift_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')