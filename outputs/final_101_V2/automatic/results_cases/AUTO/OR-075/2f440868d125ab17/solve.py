import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
if 4 in project_ids and 7 in project_ids:
    m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
else:
    raise ValueError('Project 4 or 7 not found in project.csv')
if 6 in project_ids and 1 in project_ids:
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
else:
    raise ValueError('Project 6 or 1 not found in project.csv')
if 10 in project_ids and 5 in project_ids:
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
else:
    raise ValueError('Project 10 or 5 not found in project.csv')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project {i}: {proj_name[i]} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total Capital Used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')