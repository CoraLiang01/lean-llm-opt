import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
project_ids = sorted(df['Project ID'].unique())
if len(project_ids) != 110 or min(project_ids) != 1 or max(project_ids) != 110:
    raise ValueError('Project IDs are not exactly 1..110 as required.')
try:
    npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
    capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
except Exception as e:
    raise ValueError(f'Error extracting NPV or Capital columns: {e}')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='Budget')
if 4 not in project_ids or 7 not in project_ids:
    raise ValueError('Project 4 or 7 not found in Project IDs.')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='MutEx_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise ValueError('Project 1 or 6 not found in Project IDs.')
m.addConstr(y_vars[6] <= y_vars[1], name='Prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise ValueError('Project 5 or 10 not found in Project IDs.')
m.addConstr(y_vars[10] <= y_vars[5], name='Contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            proj_row = df.loc[df['Project ID'] == i].iloc[0]
            print(f"  Project ID {i:3d}: {proj_row['Project Name']} (Capital: {capital_dict[i]} k$, NPV: {npv_dict[i]} k$)")
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')