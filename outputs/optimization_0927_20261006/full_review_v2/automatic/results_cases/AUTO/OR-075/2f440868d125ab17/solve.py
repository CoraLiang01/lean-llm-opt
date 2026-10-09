import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
name_to_id = dict(zip(df['Project Name_norm'], df['Project ID']))

def find_project_id_by_name(target_name):
    norm = target_name.strip().casefold()
    if norm not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in project.csv")
    return int(name_to_id[norm])
proj4_id = find_project_id_by_name('R&D Initiative Alpha')
proj7_id = find_project_id_by_name('Global Expansion Pilot')
proj6_id = find_project_id_by_name('System Automation')
proj1_id = find_project_id_by_name('Infrastructure Upgrade')
proj10_id = find_project_id_by_name('Customer Experience Platform')
proj5_id = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj4_id] + x_vars[proj7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj6_id] <= x_vars[proj1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[proj10_id] <= x_vars[proj5_id], name='contingent_10_requires_5')
m.optimize()