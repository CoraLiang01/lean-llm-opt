import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
row_bi = df[df['Process'].str.strip().str.casefold() == 'bi']
if row_bi.shape[0] != 1:
    raise ValueError("Expected exactly one row with Process == 'BI' in 18.csv")
task_ids = [str(i) for i in range(1, 41)]
if not all((tid in df.columns for tid in task_ids)):
    missing = [tid for tid in task_ids if tid not in df.columns]
    raise KeyError(f'Missing task columns in CSV: {missing}')
bi_values = {}
for tid in task_ids:
    val = row_bi.iloc[0][tid]
    try:
        bi_values[tid] = float(val)
    except Exception:
        raise ValueError(f'Invalid BI value for task {tid}: {val}')
cpu_ids = ['1', '2', '3']
cpu_freqs = {'1': 1.33, '2': 2.0, '3': 2.66}
m = gp.Model('TaskAssignmentMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='C_max')
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C_vars[p] == gp.quicksum((bi_values[t] / cpu_freqs[p] * x_vars[t, p] for t in task_ids)), name=f'cpu_load_{p}')
for p in cpu_ids:
    m.addConstr(C_vars[p] <= C_max_var, name=f'makespan_{p}')
m.setObjective(C_max_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {C_max_var.X:.4f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x_vars[t, p].X > 0.5]
        cpu_time = C_vars[p].X
        print(f'CPU {p} (Freq: {cpu_freqs[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {cpu_time:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')