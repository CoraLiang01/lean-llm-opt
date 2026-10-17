import gurobipy as gp
import pandas as pd
import numpy as np
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
managers = df['Unnamed: 0'].str.strip().tolist()
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
if set(managers) != set(df['Unnamed: 0'].str.strip()):
    raise ValueError('Mismatch in manager identifiers between index set and data.')
if set(projects) != set(project_cols):
    raise ValueError('Mismatch in project identifiers between index set and data.')
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost entry for manager '{m}', project '{p}'.")
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mngr, proj] * x_vars[mngr, proj] for mngr in managers for proj in projects)), gp.GRB.MINIMIZE)
for mngr in managers:
    m.addConstr(gp.quicksum((x_vars[mngr, proj] for proj in projects)) == 1, name=f'assign_mgr_{mngr}')
for proj in projects:
    m.addConstr(gp.quicksum((x_vars[mngr, proj] for mngr in managers)) == 1, name=f'assign_proj_{proj}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Optimal Assignment ---')
    for mngr in managers:
        for proj in projects:
            if x_vars[mngr, proj].X > 0.5:
                print(f'Manager {mngr} assigned to Project {proj} (Cost: {cost[mngr, proj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')