import gurobipy as gp
import pandas as pd
import numpy as np
import re
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',')
machines = cost_df['Machine'].astype(str).tolist()
tasks = [col for col in cost_df.columns if col != 'Machine']
cost = {}
for idx, row in cost_df.iterrows():
    machine = str(row['Machine'])
    for task in tasks:
        cost[machine, task] = float(row[task])
if len(machines) != 12 or len(tasks) != 12:
    raise ValueError(f'Expected 12 machines and 12 tasks, got {len(machines)} machines and {len(tasks)} tasks.')
for m in machines:
    for t in tasks:
        if (m, t) not in cost:
            raise KeyError(f'Missing cost entry for machine {m}, task {t}.')
m = gp.Model('AssignmentProblem')
x = m.addVars(machines, tasks, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in tasks)), gp.GRB.MINIMIZE)
for i in machines:
    m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == 1, name=f'assign_machine_{i}')
for j in tasks:
    m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1, name=f'assign_task_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for i in machines:
        for j in tasks:
            if x[i, j].X > 0.5:
                print(f'Machine {i} assigned to Task {j} with cost {cost[i, j]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')