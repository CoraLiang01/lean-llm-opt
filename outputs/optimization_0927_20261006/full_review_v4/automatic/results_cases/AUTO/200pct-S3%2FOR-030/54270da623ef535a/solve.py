import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df = df.set_index('Project ID', drop=False)
project_ids = df.index.tolist()
capital_dict = df['Capital (k$)'].astype(int).to_dict()
npv_dict = df['NPV (k$)'].astype(int).to_dict()
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='Budget')
if 4 not in x_vars or 7 not in x_vars:
    raise KeyError('Project 4 or 7 not found in project.csv')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='MutuallyExclusive_4_7')
if 1 not in x_vars or 6 not in x_vars:
    raise KeyError('Project 1 or 6 not found in project.csv')
m.addConstr(x_vars[6] <= x_vars[1], name='Prerequisite_6_requires_1')
if 5 not in x_vars or 10 not in x_vars:
    raise KeyError('Project 5 or 10 not found in project.csv')
m.addConstr(x_vars[10] <= x_vars[5], name='Contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.at[i, 'Project Name'] if 'Project Name' in df.columns else str(i)
            print(f'  Project {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')