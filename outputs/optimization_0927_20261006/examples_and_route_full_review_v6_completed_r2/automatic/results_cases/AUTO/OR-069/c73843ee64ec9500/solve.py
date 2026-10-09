import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
managers = df['Manager'].tolist()
project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in df.columns if project_cost_col_pattern.match(col)]
project_tuples = []
for col in project_cols:
    m = project_cost_col_pattern.match(col)
    if m:
        project_num = int(m.group(1))
        project_tuples.append((project_num, col))
project_tuples.sort()
projects = [f'Project {num}' for (num, _) in project_tuples]
project_col_by_project = {f'Project {num}': col for (num, col) in project_tuples}
c = {}
for (idx, row) in df.iterrows():
    manager = row['Manager']
    for project in projects:
        col = project_col_by_project[project]
        val = row[col].strip()
        if val == '':
            raise ValueError(f"Missing cost for manager '{manager}' and project '{project}'")
        try:
            c[manager, project] = int(val)
        except Exception as e:
            raise ValueError(f"Invalid cost value '{val}' for manager '{manager}' and project '{project}': {e}")
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((c[i, j] * x_vars[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
for i in managers:
    m.addConstr(gp.quicksum((x_vars[i, j] for j in projects)) == 1, name=f'assign_mgr_{i}')
for j in projects:
    m.addConstr(gp.quicksum((x_vars[i, j] for i in managers)) == 1, name=f'assign_proj_{j}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for i in managers:
        for j in projects:
            if x_vars[i, j].X > 0.5:
                print(f'{i} assigned to {j} (cost: {c[i, j]})')
else:
    print(f'No optimal solution found. Status: {m.status}')