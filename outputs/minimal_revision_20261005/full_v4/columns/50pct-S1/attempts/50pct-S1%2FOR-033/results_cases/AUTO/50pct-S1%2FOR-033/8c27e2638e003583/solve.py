import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others2/44.csv'
df = pd.read_csv(csv_path, sep=',')
required_columns = {'Time', 'Requirement'}
if not required_columns.issubset(df.columns):
    missing = required_columns - set(df.columns)
    raise ValueError(f'Missing required columns in CSV: {missing}')
periods = list(df['Time'])
if len(periods) != 48:
    raise ValueError(f'Expected 48 periods (half-hour slots), got {len(periods)}')
shift_starts = periods.copy()
period_idx_to_time = {i: periods[i] for i in range(len(periods))}
time_to_period_idx = {periods[i]: i for i in range(len(periods))}
requirements = df['Requirement'].astype(int).tolist()
if len(requirements) != 48:
    raise ValueError('Requirement data missing or incomplete for all periods.')
shift_length = 16
num_periods = 48

def solve_shift_scheduling():
    m = gp.Model('WaitstaffShiftScheduling')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for (t_idx, t) in enumerate(periods):
        covering_shift_starts = []
        for (s_idx, s) in enumerate(shift_starts):
            covered = [(s_idx + offset) % num_periods for offset in range(shift_length)]
            if t_idx in covered:
                covering_shift_starts.append(s)
        if not covering_shift_starts:
            raise ValueError(f'No shift starts cover period {t} (index {t_idx})')
        m.addConstr(gp.quicksum((x[s] for s in covering_shift_starts)) >= requirements[t_idx], name=f'cov_{t_idx}')
    m.optimize()
    return m
m = solve_shift_scheduling()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')