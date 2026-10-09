import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
df['Project ID'] = df['Project ID'].str.strip().astype(int)
project_ids = df['Project ID'].tolist()
if 'NPV (k$)' not in df.columns:
    raise KeyError("Required column 'NPV (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].str.strip().astype(int)))
if 'Capital (k$)' not in df.columns:
    raise KeyError("Required column 'Capital (k$)' not found in project.csv")
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].str.strip().astype(int)))
if 'Project Name' not in df.columns:
    raise KeyError("Required column 'Project Name' not found in project.csv")
project_name_dict = dict(zip(df['Project ID'], df['Project Name'].str.strip()))

def find_project_id_by_name(target_name):
    norm_target = target_name.strip().casefold()
    matches = df[df['Project Name'].str.strip().str.casefold() == norm_target]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found with name '{target_name}'")
    return int(matches.iloc[0]['Project ID'])
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_6_id = find_project_id_by_name('System Automation')
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
proj_5_id = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prerequisite_6_requires_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    total_capital = 0
    for i in project_ids:
        if x_vars[i].X > 0.5:
            print(f'  Project ID {i:3d}: {project_name_dict[i]} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            total_capital += capital_dict[i]
    print(f'Total Capital Used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')