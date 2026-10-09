import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Required Project IDs not found in project.csv: {missing_ids}')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ (Budget: 1000 k$)')
    print('\nSelected Projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | NPV: {npv_dict[i]} k$ | Capital: {capital_dict[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')