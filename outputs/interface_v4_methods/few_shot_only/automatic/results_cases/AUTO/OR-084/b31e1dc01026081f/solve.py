import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
cpus = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
task_cols = [str(i) for i in range(1, 41)]
tasks = task_cols.copy()
bi_row = df.loc[df['Process'].astype(str).str.casefold() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in the CSV.")
bi_dict = {}
for t in tasks:
    val = bi_row.iloc[0][t]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {t}')
    bi_dict[t] = float(val)
m = gp.Model('ParallelMachineMakespan')
x = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='C_max')
m.setObjective(C_max, gp.GRB.MINIMIZE)
for t in tasks:
    m.addConstr(gp.quicksum((x[t, p] for p in cpus)) == 1, name=f'assign_{t}')
for p in cpus:
    m.addConstr(C[p] == gp.quicksum((bi_dict[t] / cpu_speeds[p] * x[t, p] for t in tasks)), name=f'cpu_time_{p}')
for p in cpus:
    m.addConstr(C[p] <= C_max, name=f'makespan_{p}')
m.optimize()