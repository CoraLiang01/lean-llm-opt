import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(float)
df['NPV (k$)'] = df['NPV (k$)'].astype(float)
df = df.set_index('Project ID', drop=False)
project_ids = sorted(df.index.tolist())
if len(project_ids) != 110:
    raise ValueError('Expected 110 projects, found %d' % len(project_ids))
capital = df['Capital (k$)'].to_dict()
npv = df['NPV (k$)'].to_dict()
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in project_ids or 7 not in project_ids:
    raise KeyError('Project 4 or 7 not found in project.csv')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise KeyError('Project 1 or 6 not found in project.csv')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise KeyError('Project 5 or 10 not found in project.csv')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.at[i, 'Project Name']
            print(f'  Project {i}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
    total_capital = sum((capital[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')