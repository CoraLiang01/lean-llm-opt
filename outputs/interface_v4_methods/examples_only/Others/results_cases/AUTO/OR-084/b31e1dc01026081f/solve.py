import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
if not all((tid in df.columns for tid in task_ids)):
    missing = [tid for tid in task_ids if tid not in df.columns]
    raise ValueError(f'Missing task columns in CSV: {missing}')
row = df[df['Process'].astype(str).str.casefold() == 'bi']
if row.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
row = row.iloc[0]
task_bi = {tid: float(row[tid]) for tid in task_ids}
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, name='')
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, name='Cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((x[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C[p] == gp.quicksum((task_bi[t] / cpu_speeds[p] * x[t, p] for t in task_ids)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C[p] <= Cmax, name=f'makespan_{p}')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.optimize()