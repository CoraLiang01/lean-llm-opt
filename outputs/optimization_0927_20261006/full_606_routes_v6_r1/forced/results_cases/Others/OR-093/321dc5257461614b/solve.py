import gurobipy as gp
import pandas as pd
import numpy as np
cost_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP1/cost_12x12.csv'
cost_df = pd.read_csv(cost_path, sep=',', dtype=str, keep_default_na=False)
machines = cost_df['Machine'].tolist()
tasks = [col for col in cost_df.columns if col != 'Machine']
c_ij = {}
for (idx, row) in cost_df.iterrows():
    machine = row['Machine']
    for task in tasks:
        try:
            c_ij[machine, task] = float(row[task])
        except ValueError:
            raise ValueError(f"Invalid cost value for machine '{machine}', task '{task}': '{row[task]}'")
if len(machines) != len(tasks):
    raise ValueError(f'Number of machines ({len(machines)}) does not match number of tasks ({len(tasks)}).')
if len(machines) != 12 or len(tasks) != 12:
    raise ValueError('Expected exactly 12 machines and 12 tasks as per the query.')
m = gp.Model('AssignmentProblem')
x_vars = m.addVars(machines, tasks, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c_ij[i, j] * x_vars[i, j] for i in machines for j in tasks)), gp.GRB.MINIMIZE)
for i in machines:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in tasks)) == 1)
for j in tasks:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in machines)) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for i in machines:
        for j in tasks:
            if x_vars[i, j].X > 0.5:
                print(f'Machine {i} assigned to Task {j} with cost {c_ij[i, j]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')