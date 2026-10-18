import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
import json
csv_path = '/root/rivermind-data/lean-gptoss20b-unified-rag-20260928/source/Test_Dataset/Large-scale-or/AP_testing/AP5/manager_project_costs.csv'
df = pd.read_csv(csv_path, sep=',')
managers = df['Manager'].tolist()
project_cols = [col for col in df.columns if col.startswith('Project ') and col.endswith(' Cost')]
project_numbers = sorted([int(re.search('Project (\\d+) Cost', col).group(1)) for col in project_cols])
df_indexed = df.set_index('Manager')
cost = {(m, p): df_indexed.loc[m, f'Project {p} Cost'] for m in managers for p in project_numbers}
m = gp.Model('ManagerProjectAssignment')
x = m.addVars(managers, project_numbers, vtype=gp.GRB.BINARY)
m.setObjective(gp.quicksum((cost[m, p] * x[m, p] for m in managers for p in project_numbers)), gp.GRB.MINIMIZE)
for p in project_numbers:
    m.addConstr(gp.quicksum((x[m, p] for m in managers)) == 1)
for mng in managers:
    m.addConstr(gp.quicksum((x[mng, p] for p in project_numbers)) == 1)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total cost: {m.objVal:.2f}')
    print('--- Assignment of managers to projects ---')
    for mng in managers:
        assigned_project = None
        for p in project_numbers:
            if x[mng, p].X > 0.5:
                assigned_project = p
                break
        if assigned_project is not None:
            print(f'  Manager {mng} -> Project {assigned_project} (Cost: {cost[mng, assigned_project]})')
        else:
            print(f'  Manager {mng} -> No assignment found')
else:
    print(f'No optimal solution found. Status: {m.status}')