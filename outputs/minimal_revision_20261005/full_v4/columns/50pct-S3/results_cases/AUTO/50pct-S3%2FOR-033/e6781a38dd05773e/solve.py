import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = ['Time', 'Requirement']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
periods = list(df['Time'])
n_periods = len(periods)
if n_periods != 48:
    raise ValueError(f'Expected 48 periods, got {n_periods}')
period_idx_to_time = {i: periods[i] for i in range(n_periods)}
time_to_period_idx = {periods[i]: i for i in range(n_periods)}
requirement_series = df['Requirement']
if len(requirement_series) != n_periods:
    raise ValueError('Requirement data length does not match number of periods')
requirements = {i: int(requirement_series.iloc[i]) for i in range(n_periods)}
shift_length = 16

def solve_waitstaff_scheduling():
    m = gp.Model('WaitstaffScheduling')
    m.Params.MIPGap = 0.0001
    x = m.addVars(range(n_periods), vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[p] for p in range(n_periods))), gp.GRB.MINIMIZE)
    for t in range(n_periods):
        covering_starts = []
        for p in range(n_periods):
            covered = [(p + offset) % n_periods for offset in range(shift_length)]
            if t in covered:
                covering_starts.append(p)
        if not covering_starts:
            raise ValueError(f'No shift starts cover period {t} ({period_idx_to_time[t]})')
        m.addConstr(gp.quicksum((x[p] for p in covering_starts)) >= requirements[t], name=f'cov_{t}')
    m.optimize()
    return m
m = solve_waitstaff_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')