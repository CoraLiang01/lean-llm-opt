import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
df.columns = [col.strip() for col in df.columns]
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].to_dict()
for i in project_ids:
    if i not in npv or i not in capital:
        raise ValueError(f'Missing NPV or Capital for Project ID {i}')
proj_1 = 1
proj_4 = 4
proj_5 = 5
proj_6 = 6
proj_7 = 7
proj_10 = 10
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_1')
m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            proj_name = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {proj_name} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')