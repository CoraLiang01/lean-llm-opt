import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise ValueError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
bi_row = bi_row.iloc[0]
task_bi = {}
for t in task_ids:
    if t not in bi_row:
        raise ValueError(f"Task column '{t}' missing in CSV.")
    try:
        task_bi[t] = float(bi_row[t])
    except Exception:
        raise ValueError(f"Non-numeric or missing BI value for task '{t}': {bi_row[t]}")
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
for p in cpu_ids:
    if p not in cpu_speeds:
        raise ValueError(f"Missing GHz speed for CPU '{p}'.")
m = gp.Model('TaskAssignmentMakespan')
x_keys = [(t, p) for t in task_ids for p in cpu_ids]
x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
C_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C_vars[p] == gp.quicksum((task_bi[t] / cpu_speeds[p] * x_vars[t, p] for t in task_ids)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C_vars[p] <= Cmax, name=f'makespan_{p}')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.objVal}')
    for ((t, p), var) in x_vars.items():
        print(f'{var.VarName} {var.X}')
    for p in cpu_ids:
        print(f'{C_vars[p].VarName} {C_vars[p].X}')
    print(f'{Cmax.VarName} {Cmax.X}')
else:
    print(f'Solver status: {m.status}')