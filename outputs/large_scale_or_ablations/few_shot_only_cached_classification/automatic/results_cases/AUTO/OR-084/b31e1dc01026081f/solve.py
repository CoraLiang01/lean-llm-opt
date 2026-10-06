import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
cpus = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
task_cols = [str(i) for i in range(1, 41)]
tasks = [int(t) for t in task_cols]
bi_row = df.loc[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in the CSV.")
bi = {}
for t in tasks:
    col = str(t)
    if col not in df.columns:
        raise KeyError(f"Task column '{col}' not found in CSV.")
    val = float(bi_row.iloc[0][col])
    bi[t] = val
m = gp.Model('TaskAssignmentMakespan')
x = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
for p in cpus:
    m.addConstr(C[p] == gp.quicksum((bi[t] / cpu_speeds[p] * x[t, p] for t in tasks)), name=f'cpu_time_{p}')
m.addConstrs((C[p] <= C_max for p in cpus), name='')
m.setObjective(C_max, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpus:
        assigned_tasks = [t for t in tasks if x[t, p].X > 0.5]
        total_bi = sum((bi[t] for t in assigned_tasks))
        cpu_time = sum((bi[t] / cpu_speeds[p] for t in assigned_tasks))
        print(f'CPU {p} (speed: {cpu_speeds[p]} GHz):')
        print(f'  Assigned tasks: {assigned_tasks}')
        print(f'  Total BI: {total_bi:.2f}')
        print(f'  Completion time: {cpu_time:.4f} seconds')
    print(f'\nMakespan (C_max): {C_max.X:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')