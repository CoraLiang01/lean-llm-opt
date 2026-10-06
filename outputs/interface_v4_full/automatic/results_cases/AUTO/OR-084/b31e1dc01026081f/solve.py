import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
missing_cols = set(task_ids) - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing task columns in CSV: {missing_cols}')
row = df.iloc[0]
if not str(row['Process']).casefold().strip() == 'bi':
    raise ValueError("Expected 'BI' in the 'Process' column of the first row.")
BI = {tid: float(row[tid]) for tid in task_ids}
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
m = gp.Model('TaskAssignment_Makespan')
x = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, name='')
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, name='Cmax')
m.addConstrs((gp.quicksum((x[t, p] for p in cpu_ids)) == 1 for t in task_ids), name='')
for p in cpu_ids:
    m.addConstr(C[p] == gp.quicksum((BI[t] / cpu_speeds[p] * x[t, p] for t in task_ids)), name=f'cpu_time_{p}')
m.addConstrs((C[p] <= Cmax for p in cpu_ids), name='')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.optimize()