import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
first_col = df.columns[0]
bi_row_mask = df[first_col].str.strip().str.casefold() == 'bi'
if not bi_row_mask.any():
    raise ValueError("No row with 'BI' found in the first column of the CSV.")
bi_row = df[bi_row_mask].iloc[0]
task_cols = [col for col in df.columns if col != first_col]
bi_dict = {}
for col in task_cols:
    val = bi_row[col]
    try:
        bi = float(val)
    except Exception:
        raise ValueError(f"Non-numeric BI value '{val}' for task '{col}' in BI row.")
    bi_dict[str(col)] = bi
cpu_ids = ['1', '2', '3']
cpu_freqs = {'1': 1.33, '2': 2.0, '3': 2.66}
tasks = list(bi_dict.keys())
cpus = cpu_ids
proc_time = {}
for t in tasks:
    for p in cpus:
        proc_time[t, p] = bi_dict[t] / cpu_freqs[p]
m = gp.Model('ParallelMachineMakespan')
x_vars = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in tasks:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpus)) == 1, name=f'assign_{t}')
for p in cpus:
    m.addConstr(gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in tasks)) <= cmax_var, name=f'cpu_load_{p}')
m.setObjective(cmax_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {cmax_var.X:.6f} seconds')
    for p in cpus:
        assigned_tasks = [t for t in tasks if x_vars[t, p].X > 0.5]
        total_time = sum((proc_time[t, p] for t in assigned_tasks))
        print(f'\nCPU {p} (GHz={cpu_freqs[p]}):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.6f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')