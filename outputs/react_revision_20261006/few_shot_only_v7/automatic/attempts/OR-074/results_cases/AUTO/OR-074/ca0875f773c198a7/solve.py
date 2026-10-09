import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_min_waitstaff():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv', sep=',', dtype=str, keep_default_na=False)
    if not {'Time', 'Requirement'}.issubset(df.columns):
        raise ValueError("CSV missing required columns: 'Time', 'Requirement'")
    periods = list(df['Time'])
    n_periods = len(periods)
    if n_periods == 0:
        raise ValueError('No periods found in the input CSV.')
    period_idx_to_label = {i: periods[i] for i in range(n_periods)}
    period_label_to_idx = {periods[i]: i for i in range(n_periods)}
    try:
        requirements = df['Requirement'].astype(int).tolist()
    except Exception as e:
        raise ValueError(f"Invalid or missing values in 'Requirement' column: {e}")
    if len(requirements) != n_periods:
        raise ValueError('Mismatch between number of periods and requirements.')
    shift_length = 16
    shift_starts = list(range(n_periods))
    m = gp.Model('MinWaitstaff')
    m.setParam('MIPGap', 0.0001)
    x_vars = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x_vars[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in range(n_periods):
        covering_starts = []
        for s in shift_starts:
            covered = [(s + offset) % n_periods for offset in range(shift_length)]
            if t in covered:
                covering_starts.append(s)
        if not covering_starts:
            raise ValueError(f'No shift covers period {t} ({period_idx_to_label[t]}).')
        m.addConstr(gp.quicksum((x_vars[s] for s in covering_starts)) >= requirements[t], name=f'cov_{t}')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in shift_starts:
            print(f'x[{period_idx_to_label[s]}] {x_vars[s].VarName} {x_vars[s].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_min_waitstaff()