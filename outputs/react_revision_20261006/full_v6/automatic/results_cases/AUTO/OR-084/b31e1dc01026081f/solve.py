import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
missing_tasks = [tid for tid in task_ids if tid not in df.columns]
if missing_tasks:
    raise ValueError(f'Missing task columns in CSV: {missing_tasks}')
bi_row = df[df['Process'].str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one 'BI' row in the CSV.")
bi_row = bi_row.iloc[0]
bi_dict = {}
for tid in task_ids:
    try:
        bi_dict[tid] = float(bi_row[tid])
    except Exception as e:
        raise ValueError(f'Could not convert BI value for task {tid}: {bi_row[tid]}') from e
proc_time = {}
for tid in task_ids:
    for pid in cpu_ids:
        proc_time[tid, pid] = bi_dict[tid] / cpu_speeds[pid]
m = gp.Model('ParallelMachineMakespan')
x_vars = m.addVars([(tid, pid) for tid in task_ids for pid in cpu_ids], vtype=gp.GRB.BINARY, name='')
C_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
for tid in task_ids:
    m.addConstr(gp.quicksum((x_vars[tid, pid] for pid in cpu_ids)) == 1, name=f'assign_{tid}')
for pid in cpu_ids:
    m.addConstr(C_vars[pid] == gp.quicksum((proc_time[tid, pid] * x_vars[tid, pid] for tid in task_ids)), name=f'cpu_time_{pid}')
for pid in cpu_ids:
    m.addConstr(C_vars[pid] <= Cmax, name=f'makespan_{pid}')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')