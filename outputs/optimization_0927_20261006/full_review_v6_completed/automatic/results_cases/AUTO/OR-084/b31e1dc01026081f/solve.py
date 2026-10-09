import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
if not all((col in df.columns for col in task_ids)):
    missing = [col for col in task_ids if col not in df.columns]
    raise ValueError(f'Missing task columns in CSV: {missing}')
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
bi_row = bi_row.iloc[0]
bi_dict = {tid: float(bi_row[tid]) for tid in task_ids}
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi_dict[t] / cpu_speeds[p]
m = gp.Model('ParallelMachineMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
Cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(gp.quicksum((proc_time[t, p] * x_vars[t, p] for t in task_ids)) <= Cmax_var, name=f'makespan_cpu_{p}')
m.setObjective(Cmax_var, gp.GRB.MINIMIZE)
m.optimize()