import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_waitstaff_scheduling():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
    periods = list(df['Time'])
    if len(periods) != 48:
        raise ValueError(f'Expected 48 periods, got {len(periods)}')
    period_idx_to_time = {i: periods[i] for i in range(48)}
    time_to_period_idx = {periods[i]: i for i in range(48)}
    try:
        requirements = df['Requirement'].astype(int).to_numpy()
    except Exception as e:
        raise ValueError(f"Could not convert 'Requirement' column to int: {e}")
    if requirements.shape[0] != 48:
        raise ValueError(f'Expected 48 requirements, got {requirements.shape[0]}')
    shift_starts = list(range(48))
    m = gp.Model('WaitstaffScheduling')
    x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    shift_length = 16
    for t in range(48):
        covering_shift_starts = [s for s in shift_starts if (t - s) % 48 in range(shift_length)]
        if not covering_shift_starts:
            raise ValueError(f'No shift starts cover period {t} ({period_idx_to_time[t]})')
        m.addConstr(gp.quicksum((x_vars[s] for s in covering_shift_starts)) >= requirements[t], name=f'cover_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in shift_starts:
            print(f'{x_vars[s].VarName} {x_vars[s].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_waitstaff_scheduling()