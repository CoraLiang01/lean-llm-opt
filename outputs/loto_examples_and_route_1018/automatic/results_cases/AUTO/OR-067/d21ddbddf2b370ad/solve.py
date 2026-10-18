import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in df.columns:
    raise KeyError("Missing required column 'Unnamed: 0' in manager_project_costs.csv")
managers = df['Unnamed: 0'].apply(lambda x: x.strip()).tolist()
project_cols = [col for col in df.columns if col != 'Unnamed: 0']
projects = [col.strip() for col in project_cols]
cost = {}
for idx, row in df.iterrows():
    manager = row['Unnamed: 0'].strip()
    for project in projects:
        val = row[project]
        try:
            cost_val = int(val)
        except Exception:
            raise ValueError(f"Invalid cost value for manager '{manager}', project '{project}': '{val}'")
        cost[manager, project] = cost_val
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise KeyError(f"Missing cost entry for manager '{m}', project '{p}'")
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mng, prj] * x_vars[mng, prj] for mng in managers for prj in projects)), gp.GRB.MINIMIZE)
for mng in managers:
    m.addConstr(gp.quicksum((x_vars[mng, prj] for prj in projects)) == 1, name=f'assign_mgr_{mng}')
for prj in projects:
    m.addConstr(gp.quicksum((x_vars[mng, prj] for mng in managers)) == 1, name=f'assign_prj_{prj}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for mng in managers:
        for prj in projects:
            if x_vars[mng, prj].X > 0.5:
                print(f'Manager {mng} assigned to Project {prj} (Cost: {cost[mng, prj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')