import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture7/15.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [col for col in df.columns if col != 'Task Time Required']
tasks = task_cols
worker_id_col = 'Task Time Required'
workers = df[worker_id_col].tolist()
c = {}
for (idx, row) in df.iterrows():
    w = row[worker_id_col]
    for t in tasks:
        val = row[t]
        try:
            c[w, t] = float(val)
        except ValueError:
            raise ValueError(f"Invalid time value for worker '{w}', task '{t}': '{val}'")
m = gp.Model('WorkerTaskAssignment')
x_vars = m.addVars(workers, tasks, vtype=gp.GRB.BINARY, name='')
y_vars = m.addVars(workers, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[w, t] * x_vars[w, t] for w in workers for t in tasks)), gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[w, t] for w in workers)) == 1, name=f'assign_task_{t}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= 1, name=f'one_task_per_worker_{w}')
for w in workers:
    m.addConstr(gp.quicksum((x_vars[w, t] for t in tasks)) <= y_vars[w], name=f'link_x_y_{w}')
m.addConstr(gp.quicksum((y_vars[w] for w in workers)) == 10, name='select_10_workers')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total working hours)')
    print('\n--- Assignment Plan ---')
    for t in tasks:
        for w in workers:
            if x_vars[w, t].X > 0.5:
                print(f"Task {t}: assigned to Worker '{w}' (time: {c[w, t]})")
    print('\n--- Selected Workers ---')
    for w in workers:
        if y_vars[w].X > 0.5:
            print(f"Worker '{w}' selected")
else:
    print(f'No optimal solution found. Status: {m.status}')