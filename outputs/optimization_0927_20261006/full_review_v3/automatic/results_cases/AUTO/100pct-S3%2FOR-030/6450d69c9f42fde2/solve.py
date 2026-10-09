import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['Project ID', 'Capital (k$)', 'NPV (k$)']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
df['Project ID'] = df['Project ID'].str.strip().astype(int)
df = df.set_index('Project ID', drop=False)
df['Capital (k$)'] = df['Capital (k$)'].str.strip().astype(int)
df['NPV (k$)'] = df['NPV (k$)'].str.strip().astype(int)
project_ids = df.index.tolist()
capital_dict = df['Capital (k$)'].to_dict()
npv_dict = df['NPV (k$)'].to_dict()
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 in y_vars and 7 in y_vars:
    m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
else:
    raise ValueError('Project 4 and/or Project 7 not found in project.csv')
if 6 in y_vars and 1 in y_vars:
    m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
else:
    raise ValueError('Project 6 and/or Project 1 not found in project.csv')
if 10 in y_vars and 5 in y_vars:
    m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
else:
    raise ValueError('Project 10 and/or Project 5 not found in project.csv')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_npv = m.objVal
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    selected_projects = [i for i in project_ids if y_vars[i].X > 0.5]
    print(f'Optimal total NPV: {total_npv:.2f} k$')
    print(f'Total capital used: {total_capital:.2f} k$ (Budget: 1000 k$)')
    print(f'Number of projects selected: {len(selected_projects)}')
    print('\nSelected Projects:')
    for i in selected_projects:
        pname = df.at[i, 'Project Name'] if 'Project Name' in df.columns else str(i)
        print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')