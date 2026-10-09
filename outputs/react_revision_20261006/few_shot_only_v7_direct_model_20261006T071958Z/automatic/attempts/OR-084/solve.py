import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [str(i) for i in range(1, 41)]
for col in task_cols:
    if col not in df.columns:
        raise ValueError(f"Task column '{col}' not found in CSV.")
process_col = 'Process'
if process_col not in df.columns:
    raise ValueError(f"Column '{process_col}' not found in CSV.")
bi_row_idx = None
for (idx, val) in df[process_col].items():
    if str(val).strip().casefold() == 'bi':
        bi_row_idx = idx
        break
if bi_row_idx is None:
    raise ValueError("No row with 'Process' == 'BI' found in CSV.")
bi_row = df.loc[bi_row_idx, task_cols]
try:
    bi_dict = {t: float(bi_row[t]) for t in task_cols}
except Exception as e:
    raise ValueError(f'Failed to convert BI values to float: {e}')
cpus = ['1', '2', '3']
cpu_freq = {'1': 1.33, '2': 2.0, '3': 2.66}
if set(cpu_freq.keys()) != set(cpus):
    raise ValueError('CPU frequency keys do not match CPU identifiers.')

def solve_task_assignment(tasks, cpus, bi_dict, cpu_freq):
    m = gp.Model('TaskAssignmentMinMakespan')
    x_keys = [(t, p) for t in tasks for p in cpus]
    x_vars = m.addVars(x_keys, vtype=gp.GRB.BINARY, name='')
    C_vars = m.addVars(cpus, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
    Cmax = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
    for t in tasks:
        m.addConstr(gp.quicksum((x_vars[t, p] for p in cpus)) == 1, name=f'assign_{t}')
    for p in cpus:
        m.addConstr(C_vars[p] == gp.quicksum((bi_dict[t] / cpu_freq[p] * x_vars[t, p] for t in tasks)), name=f'cpu_time_{p}')
    for p in cpus:
        m.addConstr(C_vars[p] <= Cmax, name=f'makespan_{p}')
    m.setObjective(Cmax, gp.GRB.MINIMIZE)
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_task_assignment(task_cols, cpus, bi_dict, cpu_freq)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')