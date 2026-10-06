import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',')
managers = df['Manager'].astype(str).tolist()
project_cost_cols = [col for col in df.columns if re.match('Project\\s+\\d+\\s+Cost', col)]
projects = [re.match('(Project\\s+\\d+)\\s+Cost', col).group(1) for col in project_cost_cols]
cost = {}
for idx, row in df.iterrows():
    manager = str(row['Manager'])
    for col, project in zip(project_cost_cols, projects):
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}', project '{project}'")
        cost[manager, project] = float(val)
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for project in projects:
    m.addConstr(gp.quicksum((x[manager, project] for manager in managers)) == 1, name=f'assign_proj_{project}')
for manager in managers:
    m.addConstr(gp.quicksum((x[manager, project] for project in projects)) == 1, name=f'assign_mgr_{manager}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for manager in managers:
        for project in projects:
            if x[manager, project].X > 0.5:
                print(f"Manager '{manager}' assigned to {project} (Cost: {cost[manager, project]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')