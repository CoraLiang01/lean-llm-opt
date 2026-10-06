import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_shift_scheduling():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others2/44.csv'
    df = pd.read_csv(csv_path, sep=',')
    time_periods = list(df['Time'])
    n_periods = len(time_periods)
    if n_periods != 48:
        raise ValueError(f'Expected 48 time periods, got {n_periods}')
    period_idx_to_time = {i: time_periods[i] for i in range(n_periods)}
    time_to_period_idx = {time_periods[i]: i for i in range(n_periods)}
    if not set(['Requirement']).issubset(df.columns):
        raise KeyError("Missing 'Requirement' column in CSV")
    requirements = df['Requirement'].to_dict()
    if len(requirements) != n_periods:
        raise ValueError('Mismatch between number of periods and requirements')
    shift_length = 16
    shift_starts = list(range(n_periods))
    shift_covers = {s: [(s + k) % n_periods for k in range(shift_length)] for s in shift_starts}
    period_covered_by = {t: [] for t in range(n_periods)}
    for s in shift_starts:
        for t in shift_covers[s]:
            period_covered_by[t].append(s)
    m = gp.Model('WaitstaffShiftScheduling')
    x = m.addVars(shift_starts, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((x[s] for s in shift_starts)), gp.GRB.MINIMIZE)
    for t in range(n_periods):
        m.addConstr(gp.quicksum((x[s] for s in period_covered_by[t])) >= requirements[t], name=f'cover_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for s in shift_starts:
            print(f'x[{s}] {x[s].VarName} {x[s].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_shift_scheduling()