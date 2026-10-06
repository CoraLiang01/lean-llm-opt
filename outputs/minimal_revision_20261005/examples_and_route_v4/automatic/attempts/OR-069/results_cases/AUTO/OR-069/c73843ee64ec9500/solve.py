import gurobipy as gp
import pandas as pd
import numpy as np
costs_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv', sep=',')
managers = costs_df['Manager'].astype(str).tolist()
project_cols = [col for col in costs_df.columns if col.startswith('Project ') and col.endswith(' Cost')]
projects = [col.replace(' Cost', '') for col in project_cols]
if len(managers) != len(projects):
    raise ValueError(f'Number of managers ({len(managers)}) does not match number of projects ({len(projects)}).')
cost = {}
for (idx, row) in costs_df.iterrows():
    manager = str(row['Manager'])
    for col in project_cols:
        project = col.replace(' Cost', '')
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{manager}', project '{project}'.")
        cost[manager, project] = float(val)
for manager in managers:
    for project in projects:
        if (manager, project) not in cost:
            raise ValueError(f"Missing cost entry for manager '{manager}', project '{project}'.")

def solve_assignment_problem(managers, projects, cost):
    m = gp.Model('ManagerProjectAssignment')
    x = m.addVars([(i, j) for i in managers for j in projects], vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((cost[i, j] * x[i, j] for i in managers for j in projects)), gp.GRB.MINIMIZE)
    m.addConstrs((gp.quicksum((x[i, j] for j in projects)) == 1 for i in managers), name='')
    m.addConstrs((gp.quicksum((x[i, j] for i in managers)) == 1 for j in projects), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_assignment_problem(managers, projects, cost)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')