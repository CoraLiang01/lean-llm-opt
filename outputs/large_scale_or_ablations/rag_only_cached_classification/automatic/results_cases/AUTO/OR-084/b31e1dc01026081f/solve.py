import pandas as pd
import numpy as np
from gurobipy import Model, GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
bi_row = df.loc[df['Process'].astype(str).str.casefold().str.strip() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in the CSV.")
bi_dict = {}
for t in task_ids:
    val = bi_row.iloc[0][t]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {t}')
    bi_dict[t] = float(val)
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi_dict[t] / cpu_speeds[p]
m = Model('makespan_minimization')
assign = m.addVars(task_ids, cpu_ids, vtype=GRB.BINARY, name='')
C = m.addVars(cpu_ids, vtype=GRB.CONTINUOUS, name='')
makespan = m.addVar(vtype=GRB.CONTINUOUS, name='makespan')
for t in task_ids:
    m.addConstr(sum((assign[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C[p] == sum((proc_time[t, p] * assign[t, p] for t in task_ids)), name=f'completion_{p}')
for p in cpu_ids:
    m.addConstr(makespan >= C[p], name=f'makespan_ge_C{p}')
m.setObjective(makespan, GRB.MINIMIZE)
m.optimize()