import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP3/manager_project_costs.csv', sep=',')
managers = costs_df['Unnamed: 0'].astype(str).str.strip().tolist()
projects = [col for col in costs_df.columns if col != 'Unnamed: 0']
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) and projects ({len(projects)}) must be equal for one-to-one assignment.')
cost = {}
for (i, row) in costs_df.iterrows():
    manager = str(row['Unnamed: 0']).strip()
    for project in projects:
        val = row[project]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}' and project '{project}'.")
        cost[manager, project] = float(val)
for i in managers:
    for j in projects:
        if (i, j) not in cost:
            raise ValueError(f"Missing cost entry for manager '{i}' and project '{j}'.")

def solve_assignment_problem(managers, projects, cost):
    m = gp.Model('ManagerProjectAssignment')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
    m.optimize()
    return m
m = solve_assignment_problem(managers, projects, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')