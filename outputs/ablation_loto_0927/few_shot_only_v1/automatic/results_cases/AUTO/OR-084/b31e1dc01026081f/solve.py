import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
cpus = [1, 2, 3]
cpu_freq = {1: 1.33, 2: 2.0, 3: 2.66}
task_cols = [str(i) for i in range(1, 41)]
tasks = [int(col) for col in task_cols]
missing_cols = set(task_cols) - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing task columns in CSV: {missing_cols}')
bi_row = df.iloc[0]
task_bi = {int(col): float(bi_row[col]) for col in task_cols}
proc_time = {}
for t in tasks:
    for p in cpus:
        proc_time[t, p] = task_bi[t] / cpu_freq[p]
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, name='C_max')
m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in tasks)) for p in cpus), name='')
m.addConstrs((C[p] <= C_max for p in cpus), name='')
m.setObjective(C_max, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.6f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpus:
        assigned_tasks = [t for t in tasks if x[t, p].X > 0.5]
        total_time = sum((proc_time[t, p] for t in assigned_tasks))
        print(f'CPU {p} (GHz={cpu_freq[p]}):')
        print(f'  Assigned tasks: {assigned_tasks}')
        print(f'  Total processing time: {total_time:.6f} seconds')
    print('\n--- Per-CPU Completion Times ---')
    for p in cpus:
        print(f'CPU {p}: {C[p].X:.6f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')