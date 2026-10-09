import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', dtype=str, keep_default_na=False)
manager_col = 'Unnamed: 0'
project_cols = [col for col in costs_df.columns if col != manager_col]
managers = costs_df[manager_col].tolist()
projects = project_cols
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
cost = {}
for (i, row) in costs_df.iterrows():
    manager = row[manager_col]
    for project in projects:
        val = row[project]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
        cost[manager, project] = cost_val
for manager in managers:
    for project in projects:
        if (manager, project) not in cost:
            raise ValueError(f"Missing cost entry for manager '{manager}', project '{project}'.")
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars([(manager, project) for manager in managers for project in projects], vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[manager, project] * x_vars[manager, project] for manager in managers for project in projects)), gp.GRB.MINIMIZE)
for manager in managers:
    m.addConstr(gp.quicksum((x_vars[manager, project] for project in projects)) == 1, name='mgr')
for project in projects:
    m.addConstr(gp.quicksum((x_vars[manager, project] for manager in managers)) == 1, name='prj')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for ((manager, project), var) in x_vars.items():
        print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')