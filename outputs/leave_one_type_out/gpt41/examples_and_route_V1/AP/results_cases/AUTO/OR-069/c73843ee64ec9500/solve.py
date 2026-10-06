import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv', sep=',')
managers = costs_df['Manager'].astype(str).tolist()
project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in costs_df.columns if project_cost_col_pattern.match(col)]

def project_num(col):
    m = project_cost_col_pattern.match(col)
    return int(m.group(1)) if m else float('inf')
project_cols = sorted(project_cols, key=project_num)
projects = [f'Project {project_num(col)}' for col in project_cols]
col_to_project = {col: f'Project {project_num(col)}' for col in project_cols}
c = {}
for idx, row in costs_df.iterrows():
    manager = str(row['Manager'])
    for col in project_cols:
        project = col_to_project[col]
        cost = row[col]
        c[manager, project] = float(cost)
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
for manager in managers:
    for project in projects:
        if (manager, project) not in c:
            raise KeyError(f"Missing cost for manager '{manager}', project '{project}'.")
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for i in managers:
        for j in projects:
            if x[i, j].X > 0.5:
                print(f"Manager '{i}' assigned to {j} (Cost: {c[i, j]:.0f})")
else:
    print(f'No optimal solution found. Status: {m.status}')