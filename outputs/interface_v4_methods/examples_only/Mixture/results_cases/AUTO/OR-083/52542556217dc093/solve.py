import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [col for col in df.columns if col.strip() in list('ABCDEFGHIJ')]
if len(task_cols) != 10:
    raise ValueError("Expected exactly 10 task columns labeled 'A' to 'J'.")
workers = df['Task Time Required'].astype(str).str.strip().tolist()
if len(workers) != 12:
    raise ValueError('Expected exactly 12 workers in the data.')
tasks = [col.strip() for col in task_cols]
time = {}
for idx, w in enumerate(workers):
    for t in tasks:
        val = df.loc[idx, t]
        if pd.isnull(val):
            raise ValueError(f"Missing assignment time for worker '{w}', task '{t}'.")
        time[w, t] = float(val)

def solve_assignment_problem(workers, tasks, time):
    m = gp.Model('WorkerTaskAssignment')
    x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
    y = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((time[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    for t in tasks:
        m.addConstr(gp.quicksum((x[w, t] for w in workers)) == 1, name=f'task_{t}')
    for w in workers:
        m.addConstr(gp.quicksum((x[w, t] for t in tasks)) <= y[w], name=f'worker_task_{w}')
    m.addConstr(gp.quicksum((y[w] for w in workers)) == 10, name='select_10_workers')
    for w in workers:
        for t in tasks:
            m.addConstr(x[w, t] <= y[w], name=f'assign_valid_{w}_{t}')
    m.optimize()
    return m
m = solve_assignment_problem(workers, tasks, time)