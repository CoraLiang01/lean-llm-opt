import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
process_col = 'Process'
bi_row = df[df[process_col].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Expected exactly one row with Process == 'BI'")
bi_row = bi_row.iloc[0]
task_ids = [str(i) for i in range(1, 41)]
if not all((tid in df.columns for tid in task_ids)):
    raise ValueError('Not all task columns 1-40 found in CSV')
task_bi = {}
for tid in task_ids:
    val = bi_row[tid]
    try:
        task_bi[tid] = float(val)
    except Exception:
        raise ValueError(f'Invalid BI value for task {tid}: {val}')
cpu_ids = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
if set(cpu_ids) != set(cpu_speeds.keys()):
    raise ValueError('CPU IDs and speeds mismatch')
m = Model('task_assignment_makespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=GRB.BINARY, name='')
C_max = m.addVar(vtype=GRB.CONTINUOUS, name='C_max')
for t in task_ids:
    m.addConstr(quicksum((x_vars[t, c] for c in cpu_ids)) == 1)
for c in cpu_ids:
    m.addConstr(quicksum((task_bi[t] / cpu_speeds[c] * x_vars[t, c] for t in task_ids)) <= C_max)
m.setObjective(C_max, GRB.MINIMIZE)
m.optimize()