import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise KeyError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
bi_values = {}
for t in task_cols:
    if t not in bi_row.columns:
        raise KeyError(f"Task column '{t}' not found in CSV.")
    val = float(bi_row.iloc[0][t])
    bi_values[t] = val
cpus = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
proc_time = {}
for t in task_cols:
    proc_time[t] = {}
    for p in cpus:
        proc_time[t][p] = bi_values[t] / cpu_speeds[p]
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(task_cols, cpus, vtype=gp.GRB.BINARY, name='x')
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_cols:
    m.addConstr(gp.quicksum((x[t, p] for p in cpus)) == 1, name=f'Assign_{t}')
for p in cpus:
    m.addConstr(gp.quicksum((proc_time[t][p] * x[t, p] for t in task_cols)) <= Cmax, name=f'Makespan_CPU{p}')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.optimize()