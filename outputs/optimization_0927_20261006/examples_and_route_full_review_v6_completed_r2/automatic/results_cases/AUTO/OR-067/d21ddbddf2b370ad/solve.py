import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in df.columns:
    raise KeyError("Missing required column 'Unnamed: 0' in manager_project_costs.csv")
managers = df['Unnamed: 0'].astype(str).tolist()
project_cols = [col for col in df.columns if col != 'Unnamed: 0']
projects = project_cols.copy()
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
cost = {}
for (idx, row) in df.iterrows():
    manager = str(row['Unnamed: 0'])
    for project in projects:
        try:
            cost_val = int(row[project])
        except Exception as e:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': {row[project]}")
        cost[manager, project] = cost_val
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x_vars[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for manager in managers:
    m.addConstr(gp.quicksum((x_vars[manager, project] for project in projects)) == 1, name=f'assign_mgr_{manager}')
for project in projects:
    m.addConstr(gp.quicksum((x_vars[manager, project] for manager in managers)) == 1, name=f'assign_proj_{project}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Optimal Assignment ---')
    for manager in managers:
        for project in projects:
            if x_vars[manager, project].X > 0.5:
                print(f'Manager {manager} assigned to Project {project} (Cost: {cost[manager, project]})')
else:
    print(f'No optimal solution found. Status: {m.status}')