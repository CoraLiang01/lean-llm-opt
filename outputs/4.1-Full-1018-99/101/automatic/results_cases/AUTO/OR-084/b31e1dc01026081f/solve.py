import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise KeyError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one row with Process == 'BI'.")
bi_row = bi_row.iloc[0]
BI_t = {}
for t in task_cols:
    if t not in bi_row:
        raise KeyError(f"Missing task column '{t}' in CSV.")
    try:
        BI_t[t] = float(bi_row[t])
    except Exception as e:
        raise ValueError(f'Could not convert BI value for task {t}: {bi_row[t]}') from e
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_cols:
    proc_time[t] = {}
    for p in cpu_ids:
        proc_time[t][p] = BI_t[t] / cpu_speeds[p]
m = gp.Model('ParallelMachineScheduling_Makespan')
x_vars = m.addVars(task_cols, cpu_ids, vtype=gp.GRB.BINARY, name='')
Cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_cols:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(gp.quicksum((proc_time[t][p] * x_vars[t, p] for t in task_cols)) <= Cmax_var, name=f'makespan_cpu_{p}')
m.setObjective(Cmax_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {Cmax_var.X:.4f} seconds')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_cols if x_vars[t, p].X > 0.5]
        total_time = sum((proc_time[t][p] for t in assigned_tasks))
        print(f'\nCPU {p} (speed {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {(', '.join(assigned_tasks) if assigned_tasks else '(none)')}")
        print(f'  Total processing time: {total_time:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')