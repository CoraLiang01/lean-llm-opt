import pandas as pd
import numpy as np
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
    df = pd.read_csv(path, sep=',')
    worker_rows = df[df['Task Time Required'].astype(str).str.casefold() != 'worker'].copy()
    worker_rows['WorkerID'] = worker_rows['Task Time Required'].astype(str)
    workers = list(worker_rows['WorkerID'])
    tasks = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
    if len(workers) != 12:
        raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
    if len(tasks) != 10:
        raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
    cost = {}
    for _, row in worker_rows.iterrows():
        w = row['WorkerID']
        for t in tasks:
            val = row[t]
            if pd.isnull(val):
                raise ValueError(f'Missing time for worker {w}, task {t}')
            cost[w, t] = float(val)
    m = gp.Model('worker_task_assignment')
    assign = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
    select = m.addVars(workers, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[w, t] * assign[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((assign[w, t] for w in workers)) == 1, name=f'task_{t}_assigned_once')
    for w in workers:
        m.addConstr(gp.quicksum((assign[w, t] for t in tasks)) <= select[w], name=f'worker_{w}_assign_leq_select')
    m.addConstr(gp.quicksum((select[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(assign[w, t] <= select[w], name=f'assign_{w}_{t}_only_if_selected')
    for w in workers:
        m.addConstr(gp.quicksum((assign[w, t] for t in tasks)) <= 1, name=f'worker_{w}_at_most_one_task')
    m.optimize()
    return m
m = solve_problem()