import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if not np.issubdtype(df['Project ID'].dtype, np.integer):
    df['Project ID'] = df['Project ID'].astype(int)
project_ids = list(range(1, 111))
missing_ids = set(project_ids) - set(df['Project ID'])
if missing_ids:
    raise ValueError(f'Missing required Project IDs in CSV: {missing_ids}')
capital = df.set_index('Project ID')['Capital (k$)'].to_dict()
npv = df.set_index('Project ID')['NPV (k$)'].to_dict()
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_npv = m.objVal
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    selected_projects = [i for i in project_ids if x[i].X > 0.5]
    print(f'Optimal total NPV: {total_npv:.2f} k$')
    print(f'Total capital used: {total_capital:.2f} k$ (Budget: 1000 k$)')
    print(f'Number of selected projects: {len(selected_projects)}')
    print('Selected Projects:')
    for i in selected_projects:
        pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
        print(f'  Project {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')