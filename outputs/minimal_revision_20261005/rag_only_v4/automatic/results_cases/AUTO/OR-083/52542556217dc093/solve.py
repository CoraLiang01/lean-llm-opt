import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
    df = pd.read_csv(path, sep=',')
    caption_row_idx = df[df['Task Time Required'].astype(str).str.strip().str.casefold() == 'worker'].index
    if len(caption_row_idx) != 1:
        raise ValueError("Could not uniquely identify the caption row with 'Worker' in 'Task Time Required'.")
    caption_row_idx = caption_row_idx[0]
    worker_rows = df.loc[caption_row_idx + 1:].copy()
    worker_rows = worker_rows.dropna(how='all')
    worker_ids = worker_rows['Task Time Required'].astype(str).str.strip().tolist()
    if len(worker_ids) != 12:
        raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
    task_ids = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
    if len(task_ids) != 10:
        raise ValueError(f'Expected 10 tasks (A-J), found {len(task_ids)}: {task_ids}')
    cost = {}
    for (w_idx, w) in enumerate(worker_ids):
        row = worker_rows.iloc[w_idx]
        for t in task_ids:
            val = row[t]
            if pd.isnull(val):
                raise ValueError(f'Missing time value for worker {w}, task {t}')
            try:
                cost[w, t] = float(val)
            except Exception as e:
                raise ValueError(f'Non-numeric time value for worker {w}, task {t}: {val}')
    m = gp.Model('worker_task_assignment')
    assign = m.addVars(worker_ids, task_ids, vtype=GRB.BINARY, name='')
    select = m.addVars(worker_ids, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w, t] * assign[w, t] for w in worker_ids for t in task_ids)), GRB.MINIMIZE)
    for t in task_ids:
        m.addConstr(gp.quicksum((assign[w, t] for w in worker_ids)) == 1, name='')
    for w in worker_ids:
        m.addConstr(gp.quicksum((assign[w, t] for t in task_ids)) <= select[w], name='')
    m.addConstr(gp.quicksum((select[w] for w in worker_ids)) == 10, name='')
    for w in worker_ids:
        for t in task_ids:
            m.addConstr(assign[w, t] <= select[w], name='')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName} {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()