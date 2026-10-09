import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Manager' not in df.columns:
    raise KeyError("Required column 'Manager' not found in CSV.")
managers = df['Manager'].tolist()
project_cost_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = [col for col in df.columns if project_cost_pattern.match(col)]
if not project_cols:
    raise KeyError("No project cost columns found matching 'Project X Cost' pattern.")

def project_num(col):
    m = project_cost_pattern.match(col)
    return int(m.group(1)) if m else float('inf')
project_cols = sorted(project_cols, key=project_num)
projects = project_cols
cost = {}
for (idx, row) in df.iterrows():
    manager = row['Manager']
    for project in projects:
        val = row[project]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
        cost[manager, project] = cost_val
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}); assignment problem requires a square cost matrix.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[i, j] * x_vars[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
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
                print(f"Manager '{i}' assigned to Project '{j}' (Cost: {cost[i, j]})")
else:
    print(f'No optimal solution found. Status: {m.status}')