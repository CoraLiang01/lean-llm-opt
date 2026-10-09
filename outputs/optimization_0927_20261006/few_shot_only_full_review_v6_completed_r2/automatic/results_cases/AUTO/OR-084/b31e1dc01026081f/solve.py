import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_ids = [str(i) for i in range(1, 41)]
cpu_ids = [1, 2, 3]
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi']
if bi_row.shape[0] != 1:
    raise ValueError("Could not find exactly one row with Process == 'BI' in 18.csv")
bi_row = bi_row.iloc[0]
bi_dict = {}
for t in task_ids:
    val = bi_row[t]
    try:
        bi_dict[t] = float(val)
    except Exception:
        raise ValueError(f'Invalid BI value for task {t}: {val}')
cpu_speed_dict = {1: 1.33, 2: 2.0, 3: 2.66}
m = gp.Model('TaskAssignmentMinMakespan')
x_vars = m.addVars(task_ids, cpu_ids, vtype=gp.GRB.BINARY, name='')
C_vars = m.addVars(cpu_ids, vtype=gp.GRB.CONTINUOUS, lb=0.0, name='')
C_max_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='C_max')
for t in task_ids:
    m.addConstr(gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1, name=f'assign_{t}')
for p in cpu_ids:
    m.addConstr(C_vars[p] == gp.quicksum((bi_dict[t] / cpu_speed_dict[p] * x_vars[t, p] for t in task_ids)), name=f'cpu_time_{p}')
for p in cpu_ids:
    m.addConstr(C_vars[p] <= C_max_var, name=f'makespan_{p}')
m.setObjective(C_max_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.6f} seconds')
    print('\n--- Task Assignment ---')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_ids if x_vars[t, p].X > 0.5]
        total_time = sum((bi_dict[t] / cpu_speed_dict[p] for t in assigned_tasks))
        print(f'CPU {p} (Speed: {cpu_speed_dict[p]} GHz):')
        print(f"  Assigned tasks: {', '.join(assigned_tasks)}")
        print(f'  Total processing time: {total_time:.6f} seconds')
        print(f'  Completion time variable: {C_vars[p].X:.6f} seconds')
    print(f'\nMakespan variable (C_max): {C_max_var.X:.6f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')