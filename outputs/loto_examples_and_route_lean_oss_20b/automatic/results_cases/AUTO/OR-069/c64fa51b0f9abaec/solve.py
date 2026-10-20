import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/Users/ssyb/Documents/llm-opt/experiments/lean_oss_v20_runtime_tuning_20261015/v3_loto_20261018/project/Test_Dataset/Large-scale-or/AP_testing/AP4/manager_project_costs.csv'
df = pd.read_csv(csv_path, sep=',')
managers = df['Manager'].tolist()
project_cols = [col for col in df.columns if col.endswith('Cost')]
projects = []
for col in project_cols:
    match = re.search('Project\\s+(\\d+)\\s+Cost', col)
    if match:
        projects.append(int(match.group(1)))
projects.sort()
cost = {}
for _, row in df.iterrows():
    mgr = row['Manager']
    cost[mgr] = {}
    for proj in projects:
        col_name = f'Project {proj} Cost'
        cost[mgr][proj] = int(row[col_name])
m = gp.Model('AssignmentProblem')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY)
for mgr in managers:
    m.addConstr(x.sum(mgr, '*') == 1)
for proj in projects:
    m.addConstr(x.sum('*', proj) == 1)
m.setObjective(gp.quicksum((cost[mgr][proj] * x[mgr, proj] for mgr in managers for proj in projects)), gp.GRB.MINIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total assignment cost: {m.objVal:.0f}')
    print('\nAssignments:')
    for mgr in managers:
        for proj in projects:
            if x[mgr, proj].X > 0.5:
                print(f'  {mgr} -> Project {proj} (Cost: {cost[mgr][proj]})')
else:
    print(f'Optimization ended with status {m.status}')