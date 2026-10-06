import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_assignment_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
    df = pd.read_csv(csv_path, sep=',')
    worker_col = 'Task Time Required'
    task_cols = [col for col in df.columns if col in list('ABCDEFGHIJ')]

    def is_worker_row(row):
        val = str(row[worker_col]).strip().casefold()
        if val == '' or val == 'worker':
            return False
        return any((pd.notnull(row[tc]) for tc in task_cols))
    df_workers = df[df.apply(is_worker_row, axis=1)].copy()
    workers = [str(w).strip() for w in df_workers[worker_col]]
    if len(workers) != 12:
        raise ValueError(f'Expected 12 workers, found {len(workers)} after filtering.')
    tasks = task_cols
    if len(tasks) != 10:
        raise ValueError(f'Expected 10 tasks, found {len(tasks)}.')
    time = {}
    for idx, row in df_workers.iterrows():
        w = str(row[worker_col]).strip()
        for t in tasks:
            val = row[t]
            if pd.isnull(val):
                raise ValueError(f'Missing time for worker {w}, task {t}.')
            time[w, t] = float(val)
    m = gp.Model('WorkerTaskAssignment')
    x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((time[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= 1, name=f'worker_{w}')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) == y[w], name=f'link_{w}')
    m.optimize()
    return m
m = solve_assignment_problem()