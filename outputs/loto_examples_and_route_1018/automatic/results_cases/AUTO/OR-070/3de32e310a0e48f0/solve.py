import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Manager' not in df.columns:
    raise KeyError("Required column 'Manager' not found in manager_project_costs.csv")
managers = df['Manager'].tolist()
project_cost_col_pattern = re.compile('^Project\\s+(\\d+)\\s+Cost$', re.IGNORECASE)
project_cols = []
project_ids = []
for col in df.columns:
    m = project_cost_col_pattern.match(col.strip())
    if m:
        project_cols.append(col)
        project_ids.append(m.group(1))
if len(project_cols) == 0:
    raise KeyError('No project cost columns found in manager_project_costs.csv')
if len(managers) != len(project_cols):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(project_cols)}); assignment problem requires a square matrix.')
cost = {}
for i, manager in enumerate(managers):
    for j, (col, project_id) in enumerate(zip(project_cols, project_ids)):
        val = df.at[i, col]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value '{val}' for manager '{manager}', project '{project_id}' in column '{col}'")
        cost[manager, project_id] = cost_val
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x_vars[manager, project] for manager in managers for project in project_ids)), gp.GRB.MINIMIZE)
for project in project_ids:
    m.addConstr(gp.quicksum((x_vars[manager, project] for manager in managers)) == 1)
for manager in managers:
    m.addConstr(gp.quicksum((x_vars[manager, project] for project in project_ids)) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for manager in managers:
        for project in project_ids:
            if x_vars[manager, project].X > 0.5:
                print(f"Manager '{manager}' assigned to Project {project} (Cost: {cost[manager, project]})")
else:
    print(f'No optimal solution found. Status: {m.status}')