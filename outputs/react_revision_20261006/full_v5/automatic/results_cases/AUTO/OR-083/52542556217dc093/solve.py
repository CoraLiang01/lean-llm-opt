import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',')
worker_id_col = 'Task Time Required'
task_cols = [col for col in df.columns if col in list('ABCDEFGHIJ')]

def is_worker_label(val):
    if isinstance(val, str):
        return bool(re.fullmatch('\\s*Worker\\s*\\d+\\s*', val, re.IGNORECASE))
    return False
worker_rows = df[df[worker_id_col].apply(is_worker_label)].copy()
workers = list(worker_rows[worker_id_col].astype(str))
tasks = list(task_cols)
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
cost = {}
for (_, row) in worker_rows.iterrows():
    w = str(row[worker_id_col])
    for t in tasks:
        val = row[t]
        if pd.isnull(val):
            raise ValueError(f'Missing time value for worker {w}, task {t}')
        cost[w, t] = float(val)

def solve_assignment_problem(workers, tasks, cost):
    m = gp.Model('WorkerTaskAssignment')
    x = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
    m.addConstrs((gp.quicksum((x[w, t] for w in workers)) == 1 for t in tasks), name='')
    m.addConstrs((gp.quicksum((x[w, t] for t in tasks)) <= 1 for w in workers), name='')
    m.addConstr(gp.quicksum((x[w, t] for w in workers for t in tasks)) == 10, name='total_assignments')
    m.setObjective(gp.quicksum((cost[w, t] * x[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_assignment_problem(workers, tasks, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')