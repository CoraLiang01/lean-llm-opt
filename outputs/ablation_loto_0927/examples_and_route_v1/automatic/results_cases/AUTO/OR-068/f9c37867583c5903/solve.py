import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', sep=',')
managers = costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
cost = {}
for (idx, row) in costs_df.iterrows():
    manager = str(row['Unnamed: 0']).strip()
    for project in projects:
        cost[manager, project] = float(row[project])
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
for i in managers:
    for j in projects:
        if (i, j) not in cost:
            raise KeyError(f"Missing cost entry for manager '{i}' and project '{j}'.")
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
for i in managers:
    m.addConstr(gp.quicksum((x[i, j] for j in projects)) == 1, name=f'assign_mgr_{i}')
for j in projects:
    m.addConstr(gp.quicksum((x[i, j] for i in managers)) == 1, name=f'assign_proj_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for i in managers:
        for j in projects:
            if x[i, j].X > 0.5:
                print(f'Manager {i} assigned to Project {j} (Cost: {cost[i, j]:.0f})')
else:
    print(f'No optimal solution found. Status: {m.status}')