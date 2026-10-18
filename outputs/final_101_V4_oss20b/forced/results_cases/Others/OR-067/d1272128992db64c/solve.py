import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/AP_testing/AP2/manager_project_costs.csv'
df = pd.read_csv(path, sep=',')
df.set_index('Unnamed: 0', inplace=True)
managers = df.index.tolist()
projects = df.columns.tolist()
m = gp.Model('AssignmentProblem')
x = m.addVars(managers, projects, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((df.loc[mgr, proj] * x[mgr, proj] for mgr in managers for proj in projects)), gp.GRB.MINIMIZE)
for mgr in managers:
    m.addConstr(x.sum(mgr, '*') == 1)
for proj in projects:
    m.addConstr(x.sum('*', proj) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('Assignments:')
    for mgr in managers:
        for proj in projects:
            if x[mgr, proj].X > 0.5:
                print(f'  Manager {mgr} -> Project {proj} (cost {df.loc[mgr, proj]})')
else:
    print(f'No optimal solution found. Status: {m.status}')