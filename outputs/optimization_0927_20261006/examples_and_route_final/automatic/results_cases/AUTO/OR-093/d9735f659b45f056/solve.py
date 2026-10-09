import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
machine_ids = cost_df['Machine'].tolist()
task_ids = [col for col in cost_df.columns if col != 'Machine']
c_ij = {}
for (i, row) in cost_df.iterrows():
    machine = str(row['Machine'])
    for task in task_ids:
        try:
            c_ij[machine, task] = float(row[task])
        except ValueError:
            raise ValueError(f'Invalid cost value for machine {machine}, task {task}: {row[task]}')
if len(machine_ids) != 12 or len(task_ids) != 12:
    raise ValueError(f'Expected 12 machines and 12 tasks, got {len(machine_ids)} machines and {len(task_ids)} tasks.')
for key in [(i, j) for i in machine_ids for j in task_ids]:
    if key not in c_ij:
        raise KeyError(f'Missing cost entry for machine {key[0]}, task {key[1]}.')
m = gp.Model('AssignmentProblem')
x_vars = m.addVars(machine_ids, task_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in machine_ids for j in task_ids)), gp.GRB.MINIMIZE)
for i in machine_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in task_ids)) == 1, name=f'assign_machine_{i}')
for j in task_ids:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machine_ids)) == 1, name=f'assign_task_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for i in machine_ids:
        for j in task_ids:
            if x_vars[i, j].X > 0.5:
                print(f'Machine {i} assigned to Task {j} with cost {c_ij[i, j]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')