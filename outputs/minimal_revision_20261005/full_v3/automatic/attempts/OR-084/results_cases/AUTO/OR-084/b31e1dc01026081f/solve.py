import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_cols = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise ValueError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].astype(str).str.strip().str.casefold() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
bi_row = bi_row.iloc[0]
missing_cols = [col for col in task_cols if col not in df.columns]
if missing_cols:
    raise ValueError(f'Missing task columns in CSV: {missing_cols}')
tasks = task_cols
BI = {t: float(bi_row[t]) for t in tasks}
cpus = ['1', '2', '3']
GHz = {'1': 1.33, '2': 2.0, '3': 2.66}
for t in tasks:
    if t not in BI or not np.isfinite(BI[t]):
        raise ValueError(f'Missing or invalid BI value for task {t}.')
proc_time = {}
for t in tasks:
    for p in cpus:
        proc_time[t, p] = BI[t] / GHz[p]

def solve_problem(tasks, cpus, proc_time):
    m = gp.Model('ParallelMachineMakespan')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars([(t, p) for t in tasks for p in cpus], vtype=gp.GRB.BINARY, name='')
    C = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
    m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in tasks)) for p in cpus), name='')
    m.addConstrs((C[p] <= Cmax for p in cpus), name='')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem(tasks, cpus, proc_time)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')