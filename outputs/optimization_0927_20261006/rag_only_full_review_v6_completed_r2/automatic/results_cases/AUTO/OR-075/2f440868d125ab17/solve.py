import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    df[col] = df[col].str.strip()
    df[col] = df[col].astype(int)
project_ids = df['Project ID'].tolist()
id_to_name = dict(zip(df['Project ID'], df['Project Name']))
id_to_capital = dict(zip(df['Project ID'], df['Capital (k$)']))
id_to_npv = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(target_name):
    norm_target = target_name.strip().casefold()
    matches = df[df['Project Name'].str.strip().str.casefold() == norm_target]
    if len(matches) != 1:
        raise ValueError(f"Project name '{target_name}' matched {len(matches)} rows, expected 1.")
    return int(matches.iloc[0]['Project ID'])
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_5_id = find_project_id_by_name('Staff Training Program')
proj_6_id = find_project_id_by_name('System Automation')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
m = Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=GRB.BINARY, name='')
m.setObjective(quicksum((id_to_npv[i] * x_vars[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((id_to_capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prerequisite_6_requires_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()