import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', sep=',')
managers = costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
if costs_df.shape[0] != len(managers):
    raise ValueError('Mismatch in number of manager rows.')
if costs_df.shape[1] - 1 != len(projects):
    raise ValueError('Mismatch in number of project columns.')
cost = {}
for idx, row in costs_df.iterrows():
    m = str(row['Unnamed: 0']).strip()
    for p in projects:
        val = row[p]
        if pd.isnull(val):
            raise ValueError(f'Missing cost for manager {m}, project {p}')
        cost[m, p] = float(val)
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mng, prj] * x[mng, prj] for mng in managers for prj in projects)), gp.GRB.MINIMIZE)
for mng in managers:
    m.addConstr(gp.quicksum((x[mng, prj] for prj in projects)) == 1, name=f'assign_mgr_{mng}')
for prj in projects:
    m.addConstr(gp.quicksum((x[mng, prj] for mng in managers)) == 1, name=f'assign_prj_{prj}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Optimal Assignment ---')
    for mng in managers:
        for prj in projects:
            if x[mng, prj].X > 0.5:
                print(f'Manager {mng} assigned to Project {prj} (Cost: {cost[mng, prj]:.2f})')
else:
    print(f'No optimal solution found. Status: {m.status}')