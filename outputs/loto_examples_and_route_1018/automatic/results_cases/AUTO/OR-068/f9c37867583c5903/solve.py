import gurobipy as gp
import pandas as pd
import numpy as np
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',', dtype=str, keep_default_na=False)
if 'Unnamed: 0' not in df.columns:
    raise KeyError("Missing required column 'Unnamed: 0' in manager_project_costs.csv")
managers = df['Unnamed: 0'].astype(str).str.strip().tolist()
project_cols = [col for col in df.columns if col != 'Unnamed: 0']
projects = [col.strip() for col in project_cols]
cost_dict = {}
for idx, row in df.iterrows():
    m = str(row['Unnamed: 0']).strip()
    for p in project_cols:
        try:
            cost = int(row[p])
        except Exception as e:
            raise ValueError(f"Invalid cost value for manager '{m}', project '{p}': {row[p]}")
        cost_dict[m, p] = cost
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}); assignment problem requires equal cardinality.')
m = gp.Model('ManagerProjectAssignment')
x_vars = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost_dict[mgr, prj] * x_vars[mgr, prj] for mgr in managers for prj in projects)), gp.GRB.MINIMIZE)
for mgr in managers:
    m.addConstr(gp.quicksum((x_vars[mgr, prj] for prj in projects)) == 1, name=f'assign_mgr_{mgr}')
for prj in projects:
    m.addConstr(gp.quicksum((x_vars[mgr, prj] for mgr in managers)) == 1, name=f'assign_prj_{prj}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('--- Assignment ---')
    for mgr in managers:
        for prj in projects:
            if x_vars[mgr, prj].X > 0.5:
                print(f'Manager {mgr} assigned to Project {prj} (Cost: {cost_dict[mgr, prj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')