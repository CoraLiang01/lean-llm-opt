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
for (idx, row) in cost_df.iterrows():
    machine = str(row['Machine'])
    for task in tasks:
        if pd.isnull(row[task]):
            raise ValueError(f'Missing cost for machine {machine}, task {task}')
        cost[machine, task] = float(row[task])
for machine in machines:
    for task in tasks:
        if (machine, task) not in cost:
            raise ValueError(f'Missing cost coefficient for machine {machine}, task {task}')
m = gp.Model('AssignmentProblem')
x = m.addVars([(i, j) for i in machines for j in tasks], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in machines for j in tasks)), gp.GRB.MINIMIZE)
for i in machines:
    m.addConstr(gp.quicksum((x[i, j] for j in tasks)) == 1, name='mach')
for j in tasks:
    m.addConstr(gp.quicksum((x[i, j] for i in machines)) == 1, name='task')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in machines:
        for j in tasks:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')