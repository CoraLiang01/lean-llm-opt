import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
bi_row = bi_row.iloc[0]
task_ids = [str(i) for i in range(1, 41)]
if not all((tid in bi_row.index for tid in task_ids)):
    missing = [tid for tid in task_ids if tid not in bi_row.index]
    raise ValueError(f'Missing task columns in CSV: {missing}')
task_bi = {tid: float(bi_row[tid]) for tid in task_ids}
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = task_bi[t] / cpu_speeds[p]
m = gp.Model('TaskAssignmentMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1 for t in task_ids), name='')
m.addConstrs((gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in task_ids)) <= cmax_var for p in cpu_ids), name='')
m.setObjective(cmax_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x_vars[t, p].X > 0.5]
        total_time = sum((proc_time[t, p] for t in assigned_tasks))
        print(f'\nCPU {p} (speed: {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')