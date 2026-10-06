import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others6/18.csv'
df = pd.read_csv(csv_path, sep=',')
cpus = [1, 2, 3]
cpu_speeds = {1: 1.33, 2: 2.0, 3: 2.66}
task_cols = [str(i) for i in range(1, 41)]
tasks = task_cols.copy()
bi_row = df.loc[df['Process'].astype(str).str.casefold() == 'bi']
if bi_row.empty:
    raise ValueError("No row with Process == 'BI' found in the CSV.")
bi_dict = {}
for t in tasks:
    val = bi_row.iloc[0][t]
    if pd.isnull(val):
        raise ValueError(f'Missing BI value for task {t}.')
    bi_dict[t] = float(val)
m = gp.Model('TaskAssignmentMakespan')
x = m.addVars(tasks, cpus, vtype=gp.GRB.BINARY, name='')
Cmax = m.addVar(lb=0.0, vtype=gp.GRB.CONTINUOUS, name='Cmax')
m.setObjective(Cmax, gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[t, p] for p in cpus)) == 1 for t in tasks), name='')
for p in cpus:
    m.addConstr(gp.quicksum((bi_dict[t] / cpu_speeds[p] * x[t, p] for t in tasks)) <= Cmax, name=f'makespan_cpu{p}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal makespan (completion time of last task): {m.objVal:.4f} seconds')
    for p in cpus:
        assigned_tasks = [t for t in tasks if x[t, p].X > 0.5]
        total_time = sum((bi_dict[t] / cpu_speeds[p] for t in assigned_tasks))
        print(f'\nCPU {p} (Speed: {cpu_speeds[p]} GHz):')
        print(f"  Assigned tasks: {(', '.join(assigned_tasks) if assigned_tasks else 'None')}")
        print(f'  Total processing time: {total_time:.4f} seconds')
else:
    print(f'No optimal solution found. Status: {m.status}')