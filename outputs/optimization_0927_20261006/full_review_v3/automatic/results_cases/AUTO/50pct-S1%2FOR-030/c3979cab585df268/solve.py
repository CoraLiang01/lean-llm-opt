import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Required columns 'NPV (k$)' or 'Capital (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in x_vars or 7 not in x_vars:
    raise KeyError('Project 4 or 7 not found in project.csv')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_exclusive_4_7')
if 6 not in x_vars or 1 not in x_vars:
    raise KeyError('Project 6 or 1 not found in project.csv')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
if 10 not in x_vars or 5 not in x_vars:
    raise KeyError('Project 10 or 5 not found in project.csv')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'Project {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    print(f'\nTotal capital used: {sum((capital_dict[i] for i in selected))} k$')
    print(f'Number of projects selected: {len(selected)}')
else:
    print(f'No optimal solution found. Status: {m.status}')