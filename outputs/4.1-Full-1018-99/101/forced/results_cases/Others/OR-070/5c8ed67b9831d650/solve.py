import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Manager' not in df.columns:
    raise KeyError("Required column 'Manager' not found in manager_project_costs.csv")
managers = df['Manager'].tolist()
project_cost_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in df.columns if project_cost_pattern.match(col)]
if len(project_cols) == 0:
    raise KeyError('No project cost columns found in manager_project_costs.csv')

def project_num(col):
    m = project_cost_pattern.match(col)
    return int(m.group(1)) if m else float('inf')
project_cols = sorted(project_cols, key=project_num)
projects = [col for col in project_cols]
cost = {}
for idx, row in df.iterrows():
    manager = row['Manager']
    for project in projects:
        try:
            val = int(row[project])
        except Exception as e:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': {row[project]}")
        cost[manager, project] = val
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}); assignment problem requires a square cost matrix.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x_vars[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for project in projects:
    m.addConstr(gp.quicksum((x_vars[manager, project] for manager in managers)) == 1, name=f'assign_project_{project}')
for manager in managers:
    m.addConstr(gp.quicksum((x_vars[manager, project] for project in projects)) == 1, name=f'assign_manager_{manager}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for manager in managers:
        for project in projects:
            if x_vars[manager, project].X > 0.5:
                print(f'{manager} assigned to {project} (Cost: {cost[manager, project]})')
else:
    print(f'No optimal solution found. Status: {m.status}')