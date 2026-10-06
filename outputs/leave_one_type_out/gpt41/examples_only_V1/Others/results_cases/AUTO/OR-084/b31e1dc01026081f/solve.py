import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
bi_row = df[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in the CSV.")
bi_row = bi_row.iloc[0]
task_bi = {}
for t in task_ids:
    if t not in bi_row:
        raise KeyError(f"Task column '{t}' not found in CSV.")
    val = bi_row[t]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {t}.')
    task_bi[t] = float(val)
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.setObjective(C_max, gp.GRB.MINIMIZE)
for t in task_ids:
    m.addConstr(gp.quicksum((x[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C[p] == gp.quicksum((task_bi[t] / cpu_speeds[p] * x[t, p] for t in task_ids)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C[p] <= C_max, name=f'makespan_{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x[t, p].X > 0.5]
        total_time = sum((task_bi[t] / cpu_speeds[p] for t in assigned_tasks))
        print(f'CPU {p} (Speed: {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.4f} seconds')
    print(f'\nMakespan (C_max): {C_max.X:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')