import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Process' not in df.columns:
    raise KeyError("Column 'Process' not found in 18.csv")
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one row with Process == 'BI' in 18.csv")
bi_row = bi_row.iloc[0]
task_ids = [str(i) for i in range(1, 41)]
for t in task_ids:
    if t not in df.columns:
        raise KeyError(f"Task column '{t}' not found in 18.csv")
try:
    bi_dict = {t: float(bi_row[t]) for t in task_ids}
except Exception as e:
    raise ValueError(f'Failed to convert BI values to float: {e}')
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi_dict[t] / cpu_speeds[p]
m = gp.Model('TaskAssignmentMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
Cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in task_ids)) <= Cmax_var, name=f'cpu_load_{p}')
m.setObjective(Cmax_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last finishing task): {Cmax_var.X:.6f} seconds')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x_vars[t, p].X > 0.5]
        total_time = sum((proc_time[t, p] for t in assigned_tasks))
        print(f'\nCPU {p} (speed {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {(', '.join(assigned_tasks) if assigned_tasks else '(none)')}")
        print(f'  Total processing time: {total_time:.6f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')