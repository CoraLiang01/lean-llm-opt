import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
bi_row_idx = None
for idx, val in enumerate(df.iloc[:, 0]):
    if str(val).strip().casefold() == 'bi':
        bi_row_idx = idx
        break
if bi_row_idx is None:
    raise ValueError("Could not find 'BI' row in the CSV.")
task_ids = [col.strip() for col in df.columns[1:]]
if len(task_ids) != 40:
    raise ValueError(f'Expected 40 tasks, found {len(task_ids)} in CSV columns.')
bi_values = []
for col in df.columns[1:]:
    val = df.loc[bi_row_idx, col]
    try:
        bi = float(val)
    except Exception:
        raise ValueError(f"Could not convert BI value '{val}' for task '{col}' to float.")
    bi_values.append(bi)
tasks = [str(tid) for tid in task_ids]
if len(tasks) != 40 or len(bi_values) != 40:
    raise ValueError('Mismatch in number of tasks and BI values.')
task_bi = dict(zip(tasks, bi_values))
cpus = ['1', '2', '3']
cpu_freq = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in tasks:
    for p in cpus:
        proc_time[t, p] = task_bi[t] / cpu_freq[p]
m = gp.Model('TaskAssignmentMakespan')
x_vars = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
C_p_vars = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpus)) == 1, name=f'assign_{t}')
for p in cpus:
    m.addConstr(C_p_vars[p] == gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in tasks)), name=f'cpu_time_{p}')
for p in cpus:
    m.addConstr(C_p_vars[p] <= C_max_var, name=f'makespan_{p}')
m.setObjective(C_max_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.6f} seconds')
    print('\n--- Task Assignment ---')
    for t in tasks:
        for p in cpus:
            if x_vars[t, p].X > 0.5:
                print(f'  Task {t} (BI={task_bi[t]}) assigned to CPU {p} (GHz={cpu_freq[p]}) - Processing time: {proc_time[t, p]:.6f} s')
    print('\n--- CPU Completion Times ---')
    for p in cpus:
        print(f'  CPU {p} (GHz={cpu_freq[p]}): Completion time = {C_p_vars[p].X:.6f} s')
    print(f'\nMakespan (C_max): {C_max_var.X:.6f} s')
else:
    print(f'No optimal solution found. Status: {m.status}')