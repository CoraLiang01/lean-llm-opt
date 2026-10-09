import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
npv = df.set_index('Project ID')['NPV (k$)'].to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].to_dict()

def find_project_id_by_name(target_name):
    norm = target_name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm]['Project ID'].tolist()
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found: {matches}")
    return matches[0]
proj1_id = find_project_id_by_name('Infrastructure Upgrade')
proj4_id = find_project_id_by_name('R&D Initiative Alpha')
proj5_id = find_project_id_by_name('Staff Training Program')
proj6_id = find_project_id_by_name('System Automation')
proj7_id = find_project_id_by_name('Global Expansion Pilot')
proj10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj4_id] + x_vars[proj7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj6_id] <= x_vars[proj1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[proj10_id] <= x_vars[proj5_id], name='contingent_10_requires_5')
m.optimize()