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
bi_row = bi_row.iloc[0]
missing_cols = [col for col in task_cols if col not in df.columns]
if missing_cols:
    raise KeyError(f'Missing task columns in CSV: {missing_cols}')
tasks = task_cols
BI = {t: float(bi_row[t]) for t in tasks}
cpus = ['1', '2', '3']
GHz = {'1': 1.33, '2': 2.0, '3': 2.66}
for t in tasks:
    if not np.isfinite(BI[t]):
        raise ValueError(f'Non-finite BI value for task {t}: {BI[t]}')

def solve_problem(tasks, cpus, BI, GHz):
    m = gp.Model('ParallelMachineMakespan')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars([(t, p) for t in tasks for p in cpus], vtype=gp.GRB.BINARY, name='')
    Cmax = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='Cmax')
    m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
    m.addConstrs((gp.quicksum((BI[t] / GHz[p] * x[t, p] for t in tasks)) <= Cmax for p in cpus), name='')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem(tasks, cpus, BI, GHz)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')