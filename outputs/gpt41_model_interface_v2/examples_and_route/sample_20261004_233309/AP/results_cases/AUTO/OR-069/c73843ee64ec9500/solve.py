import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv', sep=',')
managers = costs_df['Manager'].astype(str).tolist()
project_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in costs_df.columns if project_col_pattern.match(col)]
project_cols_sorted = sorted(project_cols, key=lambda c: int(project_col_pattern.match(c).group(1)))
projects = [f'Project {project_col_pattern.match(col).group(1)}' for col in project_cols_sorted]
cost = {}
for (idx, row) in costs_df.iterrows():
    manager = str(row['Manager'])
    cost[manager] = {}
    for (col, project) in zip(project_cols_sorted, projects):
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}', project '{project}'")
        cost[manager][project] = float(val)
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager][project] * x[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for manager in managers:
    m.addConstr(gp.quicksum((x[manager, project] for project in projects)) == 1, name='')
for project in projects:
    m.addConstr(gp.quicksum((x[manager, project] for manager in managers)) == 1, name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for manager in managers:
        for project in projects:
            if x[manager, project].X > 0.5:
                print(f'{manager} assigned to {project} (Cost: {cost[manager][project]:.0f})')
else:
    print(f'No optimal solution found. Status: {m.status}')