import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
project_id_set = set(project_ids)
if len(project_id_set) != len(project_ids):
    raise ValueError('Duplicate Project IDs found in project.csv')
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(name):
    norm_name = name.strip().casefold()
    matches = df[df['Project Name'].str.strip().str.casefold() == norm_name]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found with name '{name}'")
    return int(matches.iloc[0]['Project ID'])
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_5_id = find_project_id_by_name('Staff Training Program')
proj_6_id = find_project_id_by_name('System Automation')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()