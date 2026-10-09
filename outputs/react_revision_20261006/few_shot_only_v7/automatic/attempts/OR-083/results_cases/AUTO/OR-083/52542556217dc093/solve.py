import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_assignment_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
    df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
    task_col = 'Task Time Required'
    worker_cols = [col for col in df.columns if col != task_col]
    tasks = df[task_col].tolist()
    workers = worker_cols
    if len(tasks) != 10:
        raise ValueError(f'Expected 10 tasks, found {len(tasks)} in CSV.')
    if len(workers) != 12:
        raise ValueError(f'Expected 12 workers, found {len(workers)} in CSV.')
    time = {}
    for (t_idx, t) in enumerate(tasks):
        for w in workers:
            val = df.loc[t_idx, w]
            try:
                time[w, t] = float(val)
            except Exception:
                raise ValueError(f"Invalid or missing time value for worker '{w}', task '{t}': '{val}'")
    m = gp.Model('WorkerTaskAssignment')
    x_keys = [(w, t) for w in workers for t in tasks]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((time[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'task_{t}_assign')
    for w in workers:
        m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_{w}_assign')
    m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(x_vars[w, t] <= y_vars[w], name=f'assign_lim_{w}_{t}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_assignment_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')