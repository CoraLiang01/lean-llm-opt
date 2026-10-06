import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv', sep=',')
if 'Unnamed: 0' not in costs_df.columns:
    raise KeyError("Missing required column 'Unnamed: 0' in manager_project_costs.csv")
managers = costs_df['Unnamed: 0'].astype(str).tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
if costs_df.shape != (len(managers), len(projects) + 1):
    raise ValueError('Mismatch in cost matrix dimensions: expected {} managers and {} projects, got shape {}'.format(len(managers), len(projects), costs_df.shape))
cost = {}
for (idx, row) in costs_df.iterrows():
    m = str(row['Unnamed: 0'])
    for p in projects:
        if pd.isnull(row[p]):
            raise ValueError(f"Missing cost for manager '{m}' and project '{p}'")
        cost[m, p] = float(row[p])
for m in managers:
    for p in projects:
        if (m, p) not in cost:
            raise ValueError(f"Missing cost entry for manager '{m}' and project '{p}'")

def solve_assignment_problem(managers, projects, cost):
    m = gp.Model('ManagerProjectAssignment')
    x = m.addVars([(m, p) for m in managers for p in projects], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[m, p] * x[m, p] for m in managers for p in projects)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[m, p] for p in projects)) == 1 for m in managers), name='')
    m.addConstrs((gp.quicksum((x[m, p] for m in managers)) == 1 for p in projects), name='')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_assignment_problem(managers, projects, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')