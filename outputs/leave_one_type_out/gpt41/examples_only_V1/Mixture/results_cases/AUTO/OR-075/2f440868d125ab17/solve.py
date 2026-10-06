import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
capital = df.set_index('Project ID')['Capital (k$)'].astype(int).to_dict()
npv = df.set_index('Project ID')['NPV (k$)'].astype(int).to_dict()
required_projects = [1, 4, 5, 6, 7, 10]
missing = [pid for pid in required_projects if pid not in project_ids]
if missing:
    raise ValueError(f'Required project IDs for constraints not found in data: {missing}')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_requires_1')
m.addConstr(x[10] <= x[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_npv = m.objVal
    total_capital = sum((capital[i] * x[i].X for i in project_ids))
    selected_projects = [i for i in project_ids if x[i].X > 0.5]
    print(f'Optimal total NPV: {total_npv:.2f} k$')
    print(f'Total capital used: {total_capital:.2f} k$')
    print(f'Number of projects selected: {len(selected_projects)}')
    print('\nSelected Projects:')
    for i in selected_projects:
        pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
        print(f'  Project ID {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')