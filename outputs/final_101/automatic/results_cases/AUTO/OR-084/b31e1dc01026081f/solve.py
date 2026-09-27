import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [str(i) for i in range(1, 41)]
if not all((col in df.columns for col in task_cols)):
    missing = [col for col in task_cols if col not in df.columns]
    raise KeyError(f'Missing task columns in CSV: {missing}')
row_bi = df.loc[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if row_bi.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
task_sizes = {t: float(row_bi.iloc[0][t]) for t in task_cols}
cpus = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(task_cols, cpus, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in task_cols), name='')
for p in cpus:
    m.addConstr(C[p] == gp.quicksum((task_sizes[t] / cpu_speeds[p] * x[t, p] for t in task_cols)), name=f'cpu_time_{p}')
m.addConstrs((C[p] <= C_max for p in cpus), name='')
m.setObjective(C_max, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpus:
        assigned_tasks = [t for t in task_cols if x[t, p].X > 0.5]
        total_time = sum((task_sizes[t] / cpu_speeds[p] for t in assigned_tasks))
        print(f'CPU {p} (speed {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')