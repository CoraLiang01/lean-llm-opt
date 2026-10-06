import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Project ID'].duplicated().any():
    raise ValueError('Duplicate Project IDs found in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df = df.set_index('Project ID', drop=False)
project_ids = df.index.tolist()
if df['NPV (k$)'].isnull().any() or df['Capital (k$)'].isnull().any():
    raise ValueError('Missing NPV or Capital values in project.csv')
npv = df['NPV (k$)'].astype(float).to_dict()
capital = df['Capital (k$)'].astype(float).to_dict()
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.at[i, 'Project Name']
            print(f'  Project ID {i}: {pname} (Capital: {capital[i]:.0f} k$, NPV: {npv[i]:.0f} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')