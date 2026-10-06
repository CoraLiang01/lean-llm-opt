import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', sep=',')
manager_col = 'Unnamed: 0'
managers = costs_df[manager_col].astype(str).tolist()
projects = [col for col in costs_df.columns if col != manager_col]
cost = {}
for idx, row in costs_df.iterrows():
    manager = str(row[manager_col])
    for project in projects:
        cost[manager, project] = float(row[project])
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
for manager in managers:
    for project in projects:
        if (manager, project) not in cost:
            raise KeyError(f"Missing cost entry for manager '{manager}' and project '{project}'.")
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for manager in managers:
    m.addConstr(gp.quicksum((x[manager, project] for project in projects)) == 1, name=f'assign_mgr_{manager}')
for project in projects:
    m.addConstr(gp.quicksum((x[manager, project] for manager in managers)) == 1, name=f'assign_proj_{project}')
m.optimize()