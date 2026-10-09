import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise KeyError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].astype(str).str.strip().str.casefold() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
bi_row = bi_row.iloc[0]
bi_dict = {}
for t in task_cols:
    if t not in bi_row:
        raise KeyError(f"Task column '{t}' missing in CSV.")
    val = bi_row[t]
    if pd.isnull(val):
        raise ValueError(f"Missing BI value for task '{t}'.")
    bi_dict[t] = float(val)
cpus = ['1', '2', '3']
cpu_speed = {'1': 1.33, '2': 2.0, '3': 2.66}
for p in cpus:
    if p not in cpu_speed:
        raise KeyError(f"CPU '{p}' missing GHz specification.")
proc_time = {}
for t in task_cols:
    for p in cpus:
        proc_time[t, p] = bi_dict[t] / cpu_speed[p]
m = gp.Model('TaskAssignmentMakespan')
x = m.addVars(task_cols, cpus, vtype=gp.GRB.BINARY, name='')
C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in task_cols), name='')
m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in task_cols)) for p in cpus), name='')
m.addConstrs((C[p] <= C_max for p in cpus), name='')
m.setObjective(C_max, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')