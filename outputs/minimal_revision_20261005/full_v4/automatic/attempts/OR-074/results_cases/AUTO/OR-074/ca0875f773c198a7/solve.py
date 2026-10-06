import gurobipy as gp
import pandas as pd
import numpy as np
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
if not {'Time', 'Requirement'}.issubset(df.columns):
    raise ValueError("CSV must contain 'Time' and 'Requirement' columns.")
periods = list(df['Time'])
requirements = df['Requirement'].astype(int).tolist()
if len(periods) != 48:
    raise ValueError('Expected 48 half-hour periods in the day.')
period_idx_to_time = {i: periods[i] for i in range(48)}
time_to_period_idx = {periods[i]: i for i in range(48)}
shift_length = 16
num_periods = 48

def solve_minimum_waitstaff(periods, requirements, shift_length):
    m = gp.Model('MinimumWaitstaff')
    m.Params.MIPGap = 0.0001
    period_indices = list(range(len(periods)))
    x = m.addVars(period_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[t] for t in period_indices)), gp.GRB.MINIMIZE)
    for p in period_indices:
        covering_starts = []
        for s in period_indices:
            if (p - s) % num_periods < shift_length:
                covering_starts.append(s)
        if not covering_starts:
            raise ValueError(f'No shift starts cover period {p} ({periods[p]})')
        m.addConstr(gp.quicksum((x[s] for s in covering_starts)) >= requirements[p], name=f'cov_{p}')
    return m
m = solve_minimum_waitstaff(periods, requirements, shift_length)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')