import gurobipy as gp
import pandas as pd
import numpy as np
import re
costs_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(costs_path, sep=',')
managers = df['Manager'].astype(str).tolist()
project_cost_cols = [col for col in df.columns if re.match('Project \\d+ Cost$', col)]
projects = [re.match('(Project \\d+) Cost$', col).group(1) for col in project_cost_cols]
cost = {}
for idx, row in df.iterrows():
    m = str(row['Manager'])
    cost[m] = {}
    for col, p in zip(project_cost_cols, projects):
        val = row[col]
        if pd.isnull(val):
            raise ValueError(f"Missing cost for manager '{m}', project '{p}'")
        cost[m][p] = float(val)
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((cost[mgr][prj] * x[mgr, prj] for mgr in managers for prj in projects)), gp.GRB.MINIMIZE)
for prj in projects:
    m.addConstr(gp.quicksum((x[mgr, prj] for mgr in managers)) == 1, name=f'assign_{prj}')
for mgr in managers:
    m.addConstr(gp.quicksum((x[mgr, prj] for prj in projects)) == 1, name=f'assign_{mgr}')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.2f}')
    print('--- Assignment ---')
    for mgr in managers:
        for prj in projects:
            if x[mgr, prj].X > 0.5:
                print(f"Manager '{mgr}' assigned to {prj} (Cost: {cost[mgr][prj]:.2f})")
else:
    print(f'No optimal solution found. Status: {m.status}')