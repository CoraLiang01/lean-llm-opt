import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)}.')
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()

def find_project_id_by_name(name):
    matches = df[df['Project Name'].str.casefold().str.strip() == name.casefold().strip()]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_6_id = find_project_id_by_name('System Automation')
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
proj_5_id = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_requires_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_requires_5')
m.optimize()