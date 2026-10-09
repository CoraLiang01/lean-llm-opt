import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))

def find_project_id_by_name(target_name):
    norm_target = target_name.strip().casefold()
    matches = df[df['Project Name'].str.strip().str.casefold() == norm_target]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_6_id = find_project_id_by_name('System Automation')
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
proj_5_id = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj_4_id] + y_vars[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[proj_6_id] <= y_vars[proj_1_id], name='prereq_6_requires_1')
m.addConstr(y_vars[proj_10_id] <= y_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')