import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', sep=',')
managers = costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
cost = {}
for idx, row in costs_df.iterrows():
    m = str(row['Unnamed: 0']).strip()
    for p in projects:
        cost[m, p] = float(row[p])
if set(managers) != set(costs_df['Unnamed: 0'].astype(str).str.strip()):
    raise ValueError('Mismatch in manager identifiers between index set and data.')
if set(projects) != set(costs_df.columns) - {'Unnamed: 0'}:
    raise ValueError('Mismatch in project identifiers between index set and data.')
if len(cost) != len(managers) * len(projects):
    raise ValueError('Cost matrix is incomplete.')
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr, prj] * x[mgr, prj] for mgr in managers for prj in projects)), gp.GRB.MINIMIZE)
for mgr in managers:
    m.addConstr(gp.quicksum((x[mgr, prj] for prj in projects)) == 1, name=f'mgr_{mgr}')
for prj in projects:
    m.addConstr(gp.quicksum((x[mgr, prj] for mgr in managers)) == 1, name=f'prj_{prj}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Assignment ---')
    for mgr in managers:
        for prj in projects:
            if x[mgr, prj].X > 0.5:
                print(f'Manager {mgr} assigned to Project {prj} (Cost: {cost[mgr, prj]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')