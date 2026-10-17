import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
expected_task_labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
task_labels = [col for col in df.columns if col.strip() in expected_task_labels]
if len(task_labels) != 10:
    raise ValueError(f'Expected 10 task columns labeled A-J, found: {task_labels}')
worker_id_col = df.columns[0]
worker_ids = df[worker_id_col].tolist()
if len(worker_ids) != 12:
    raise ValueError(f'Expected 12 workers, found {len(worker_ids)}: {worker_ids}')
time_required = {}
for idx, row in df.iterrows():
    w = str(row[worker_id_col]).strip()
    for t in task_labels:
        cell = row[t]
        try:
            time = float(cell)
        except Exception:
            raise ValueError(f"Invalid time value for worker {w}, task {t}: '{cell}'")
        time_required[w, t] = time
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(worker_ids, task_labels, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(worker_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((time_required[w, t] * x_vars[w, t] for w in worker_ids for t in task_labels)), gp.GRB.MINIMIZE)
for t in task_labels:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in worker_ids)) == 1, name=f'task_{t}_assigned_once')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_labels)) <= 1, name=f'worker_{w}_at_most_one_task')
for w in worker_ids:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in task_labels)) <= y_vars[w], name=f'worker_{w}_link_y')
m.addConstr(gp.quicksum((y_vars[w] for w in worker_ids)) == 10, name='select_10_workers')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total working hours)')
    print('\n--- Assignment Plan ---')
    for t in task_labels:
        for w in worker_ids:
            if x_vars[w, t].X > 0.5:
                print(f'Task {t}: assigned to Worker {w} (time: {time_required[w, t]})')
    print('\n--- Selected Workers ---')
    for w in worker_ids:
        if y_vars[w].X > 0.5:
            print(f'Worker {w}: SELECTED')
else:
    print(f'No optimal solution found. Status: {m.status}')