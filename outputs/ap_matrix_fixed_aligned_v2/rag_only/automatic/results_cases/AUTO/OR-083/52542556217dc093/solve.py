import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
task_columns = [col for col in df.columns if col != 'Task Time Required']
df = df[~df['Task Time Required'].str.strip().str.casefold().eq('worker')].copy()
workers = df['Task Time Required'].tolist()
tasks = task_columns
if len(workers) != 12:
    raise ValueError(f'Expected 12 workers, found {len(workers)}: {workers}')
if len(tasks) != 10:
    raise ValueError(f'Expected 10 tasks, found {len(tasks)}: {tasks}')
cost = {}
for _, row in df.iterrows():
    w = row['Task Time Required']
    for t in tasks:
        val = row[t]
        try:
            cost[w, t] = float(val)
        except Exception as e:
            raise ValueError(f"Invalid time value for worker '{w}', task '{t}': '{val}'") from e
m = Model('worker_task_assignment')
x_vars = m.addVars(workers, tasks, vtype=GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=GRB.BINARY, name='')
for t in tasks:
    m.addConstr(quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_task_{t}')
for w in workers:
    m.addConstr(quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'worker_task_limit_{w}')
m.addConstr(quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
for w in workers:
    for t in tasks:
        m.addConstr(x_vars[w, t] <= y_vars[w], name=f'assign_only_if_selected_{w}_{t}')
for w in workers:
    m.addConstr(quicksum((x_vars[w, t] for t in tasks)) <= 1, name=f'one_task_per_worker_{w}')
m.setObjective(quicksum((cost[w, t] * x_vars[w, t] for w in workers for t in tasks)), GRB.MINIMIZE)
m.optimize()