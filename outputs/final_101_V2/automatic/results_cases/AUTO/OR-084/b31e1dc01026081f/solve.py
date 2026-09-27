import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
bi_row = df.loc[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if bi_row.empty:
    raise ValueError("No row with 'Process' == 'BI' found in the CSV.")
bi_row = bi_row.iloc[0]
bi = {}
for t in task_ids:
    if t not in bi_row:
        raise KeyError(f"Task column '{t}' not found in CSV.")
    val = bi_row[t]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {t}.')
    bi[t] = float(val)
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi[t] / cpu_speeds[p]
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x[t, p] for p in cpu_ids)) == 1 for t in task_ids), name='')
m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in task_ids)) for p in cpu_ids), name='')
m.addConstrs((C[p] <= C_max for p in cpu_ids), name='')
m.setObjective(C_max, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x[t, p].X > 0.5]
        total_time = sum((proc_time[t, p] for t in assigned_tasks))
        print(f'CPU {p} (speed {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.4f} seconds')
    print(f'\nMakespan (C_max): {C_max.X:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')