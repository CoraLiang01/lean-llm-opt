import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
task_cols = [str(i) for i in range(1, 41)]
if not any(df['Process'].str.strip().str.casefold() == 'bi'):
    raise ValueError("No row with Process == 'BI' found in CSV.")
bi_row = df[df['Process'].str.strip().str.casefold() == 'bi'].iloc[0]
bi_dict = {}
for t in task_cols:
    try:
        bi_dict[t] = float(bi_row[t])
    except Exception as e:
        raise ValueError(f'Could not convert BI value for task {t}: {bi_row[t]}') from e
cpu_ids = ['1', '2', '3']
cpu_speeds = {'1': 1.33, '2': 2.0, '3': 2.66}
m = gp.Model('ParallelMachineMakespan')
x_vars = m.addVars(task_cols, cpu_ids, vtype=gp.GRB.BINARY, name='')
Cmax_var = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0.0, name='Cmax')
m.addConstrs((gp.quicksum((x_vars[t, p] for p in cpu_ids)) == 1 for t in task_cols), name='')
for p in cpu_ids:
    m.addConstr(gp.quicksum((bi_dict[t] / cpu_speeds[p] * x_vars[t, p] for t in task_cols)) <= Cmax_var, name=f'CPUload_{p}')
m.setObjective(Cmax_var, gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {Cmax_var.X:.4f} seconds')
    for p in cpu_ids:
        assigned_tasks = [t for t in task_cols if x_vars[t, p].X > 0.5]
        total_load = sum((bi_dict[t] / cpu_speeds[p] for t in assigned_tasks))
        print(f'\nCPU {p} (speed {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {(', '.join(assigned_tasks) if assigned_tasks else '(none)')}")
        print(f'  Total processing time: {total_load:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')