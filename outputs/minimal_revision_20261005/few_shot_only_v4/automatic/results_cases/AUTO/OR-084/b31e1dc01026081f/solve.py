import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
    df = pd.read_csv(csv_path, sep=',')
    task_cols = [str(i) for i in range(1, 41)]
    for col in task_cols:
        if col not in df.columns:
            raise ValueError(f"Task column '{col}' not found in CSV.")
    tasks = task_cols
    cpus = ['1', '2', '3']
    if 'Process' not in df.columns:
        raise ValueError("Column 'Process' not found in CSV.")
    bi_row = df[df['Process'].astype(str).str.casefold() == 'bi']
    if bi_row.shape[0] != 1:
        raise ValueError("Could not find exactly one 'BI' row in the CSV.")
    bi_row = bi_row.iloc[0]
    bi = {}
    for t in tasks:
        try:
            val = float(bi_row[t])
        except Exception:
            raise ValueError(f"Missing or invalid BI value for task '{t}'.")
        bi[t] = val
    cpu_freq = {'1': 1.33, '2': 2.0, '3': 2.66}
    proc_time = {}
    for t in tasks:
        for p in cpus:
            proc_time[t, p] = bi[t] / cpu_freq[p]
    m = gp.Model('TaskAssignmentMakespan')
    x = m.addVars([(t, p) for t in tasks for p in cpus], vtype=gp.GRB.BINARY, name='')
    C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='C_max')
    m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
    m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in tasks)) for p in cpus), name='')
    m.addConstrs((C[p] <= C_max for p in cpus), name='')
    m.setObjective(C_max, gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal {m.objVal}')
        for var in m.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()