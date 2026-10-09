import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
worker_id_col = 'Task Time Required'
worker_ids = []
for val in df[worker_id_col]:
    try:
        v = int(val)
        if 1 <= v <= 12:
            worker_ids.append(str(v))
    except Exception:
        continue
worker_ids = [str(w) for w in worker_ids]
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
task_labels = [col for col in df.columns if re.fullmatch('[A-J]', col)]
if len(task_labels) != 10:
    raise ValueError(f'Expected 10 tasks (A-J), found {len(task_labels)}: {task_labels}')
cost = {}
df_workers = df[df[worker_id_col].isin(worker_ids)].copy()
df_workers = df_workers.set_index(worker_id_col)
for w in worker_ids:
    for t in task_labels:
        val = df_workers.at[w, t]
        try:
            cost[w, t] = float(val)
        except Exception:
            raise ValueError(f"Invalid or missing time for worker {w}, task {t}: '{val}'")
if len(cost) != 12 * 10:
    raise ValueError(f'Cost matrix incomplete: expected 120 entries, got {len(cost)}')
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars([(w, t) for w in worker_ids for t in task_labels], vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[w, t] * x_vars[w, t] for w in worker_ids for t in task_labels)), gp.GRB.MINIMIZE)
for t in task_labels:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name='task_' + t)
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_labels)) <= y_vars[w], name='worker_' + w + '_assign')
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids for t in task_labels)) == 10, name='assign_10_tasks')
for w in worker_ids:
    for t in task_labels:
        m.addConstr(x_vars[w, t] <= y_vars[w], name='link_' + w + '_' + t)
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for w in worker_ids:
        print(f'y[{w}] {y_vars[w].VarName} {y_vars[w].X}')
    for w in worker_ids:
        for t in task_labels:
            print(f'x[{w},{t}] {x_vars[w, t].VarName} {x_vars[w, t].X}')
else:
    print(f'Solver status: {m.status}')