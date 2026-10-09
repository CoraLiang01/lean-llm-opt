import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column 'Project ID' in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Missing required columns 'NPV (k$)' or 'Capital (k$)' in project.csv")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='Budget')
if 4 not in project_ids or 7 not in project_ids:
    raise KeyError('Project 4 or Project 7 not found in project.csv')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='MutualExcl_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise KeyError('Project 1 or Project 6 not found in project.csv')
m.addConstr(x_vars[6] <= x_vars[1], name='Prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise KeyError('Project 5 or Project 10 not found in project.csv')
m.addConstr(x_vars[10] <= x_vars[5], name='Contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            proj_row = df[df['Project ID'].astype(int) == i].iloc[0]
            print(f"Project ID: {i:3d} | Name: {proj_row['Project Name']} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$")
else:
    print(f'No optimal solution found. Status: {m.status}')