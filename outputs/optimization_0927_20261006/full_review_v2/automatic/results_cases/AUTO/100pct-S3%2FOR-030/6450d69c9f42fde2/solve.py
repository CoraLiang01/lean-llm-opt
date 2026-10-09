import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)', 'Project Name'}.issubset(df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(df, name):
    norm_name = name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm_name]
    if len(matches) == 0:
        raise ValueError(f"Project name '{name}' not found in project.csv")
    return int(matches.iloc[0]['Project ID'])
special_projects = {1: 'Infrastructure Upgrade', 4: 'R&D Initiative Alpha', 5: 'Staff Training Program', 6: 'System Automation', 7: 'Global Expansion Pilot', 10: 'Customer Experience Platform'}
for (pid, pname) in special_projects.items():
    actual_pid = find_project_id_by_name(df, pname)
    if actual_pid != pid:
        raise ValueError(f"Project ID {pid} does not match expected name '{pname}' (found ID {actual_pid})")
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_requires_1')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
    total_capital = sum((capital[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')