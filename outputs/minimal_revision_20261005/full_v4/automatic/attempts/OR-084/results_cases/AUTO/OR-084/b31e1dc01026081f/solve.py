import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
task_ids = [str(i) for i in range(1, 41)]
if 'Process' not in df.columns:
    raise KeyError("Missing 'Process' column in CSV.")
bi_row = df[df['Process'].str.casefold() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in CSV.")
bi_dict = {}
for t in task_ids:
    if t not in bi_row.columns:
        raise KeyError(f"Task column '{t}' not found in CSV.")
    val = bi_row.iloc[0][t]
    if pd.isnull(val):
        raise ValueError(f"Missing BI value for task '{t}'.")
    bi_dict[t] = float(val)
cpu_ids = [1, 2, 3]
cpu_ghz = {1: 1.33, 2: 2.0, 3: 2.66}
if len(bi_dict) != 40:
    raise ValueError('Did not find 40 BI values for tasks 1-40.')
proc_time = {}
for t in task_ids:
    for p in cpu_ids:
        proc_time[t, p] = bi_dict[t] / cpu_ghz[p]

def solve_problem():
    m = gp.Model('ParallelMachineMakespan')
    m.Params.MIPGap = 0.0001
    x = m.addVars([(t, p) for t in task_ids for p in cpu_ids], vtype=gp.GRB.BINARY, name='')
    C = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    m.addConstrs((gp.quicksum((x[t, p] for p in cpu_ids)) == 1 for t in task_ids), name='')
    m.addConstrs((C[p] == gp.quicksum((proc_time[t, p] * x[t, p] for t in task_ids)) for p in cpu_ids), name='')
    m.addConstrs((C[p] <= Cmax for p in cpu_ids), name='')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')