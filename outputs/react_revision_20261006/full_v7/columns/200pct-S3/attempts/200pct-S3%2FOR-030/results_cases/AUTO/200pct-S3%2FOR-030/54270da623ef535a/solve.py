import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
try:
    npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
    capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting NPV or Capital columns to int: {e}')
if set(project_ids) != set(npv_dict.keys()) or set(project_ids) != set(capital_dict.keys()):
    raise ValueError('Mismatch in Project IDs and NPV/Capital data.')

def find_project_id_by_name(target_name):
    norm_target = target_name.casefold().strip()
    matches = df[df['Project Name'].str.casefold().str.strip() == norm_target]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
special_projects = {'R&D Initiative Alpha': None, 'Global Expansion Pilot': None, 'System Automation': None, 'Infrastructure Upgrade': None, 'Customer Experience Platform': None, 'Staff Training Program': None}
for name in special_projects:
    special_projects[name] = find_project_id_by_name(name)
proj_4 = special_projects['R&D Initiative Alpha']
proj_7 = special_projects['Global Expansion Pilot']
proj_6 = special_projects['System Automation']
proj_1 = special_projects['Infrastructure Upgrade']
proj_10 = special_projects['Customer Experience Platform']
proj_5 = special_projects['Staff Training Program']

def solve_project_selection(project_ids, npv_dict, capital_dict, proj_4, proj_7, proj_6, proj_1, proj_10, proj_5):
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((x_vars[i] * npv_dict[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x_vars[i] * capital_dict[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_1')
    m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_project_selection(project_ids=project_ids, npv_dict=npv_dict, capital_dict=capital_dict, proj_4=proj_4, proj_7=proj_7, proj_6=proj_6, proj_1=proj_1, proj_10=proj_10, proj_5=proj_5)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')