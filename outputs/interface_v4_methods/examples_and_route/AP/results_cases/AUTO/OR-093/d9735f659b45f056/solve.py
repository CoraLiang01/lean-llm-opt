import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',')
machines = cost_df['Machine'].astype(str).tolist()
tasks = [col for col in cost_df.columns if col != 'Machine']
if len(machines) != 12 or len(tasks) != 12:
    raise ValueError(f'Expected 12 machines and 12 tasks, got {len(machines)} machines and {len(tasks)} tasks.')
cost = {}
for idx, row in cost_df.iterrows():
    machine = str(row['Machine'])
    for task in tasks:
        cost[machine, task] = float(row[task])
for i in machines:
    for j in tasks:
        if (i, j) not in cost:
            raise KeyError(f'Missing cost coefficient for machine {i}, task {j}')
m = gp.Model('AssignmentProblem')
x = m.addVars(machines, tasks, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in tasks)), gp.GRB.MINIMIZE)
for i in machines:
    m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == 1, name=f'assign_machine_{i}')
for j in tasks:
    m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1, name=f'assign_task_{j}')
m.optimize()