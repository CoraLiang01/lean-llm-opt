import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if not set(['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']).issubset(df.columns):
    raise ValueError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got %d' % len(set(project_ids)))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
name_to_id = {n.casefold().strip(): pid for (n, pid) in zip(df['Project Name'], df['Project ID'])}

def find_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in Project Name column")
    return name_to_id[key]
proj_1 = find_project_id_by_name('Infrastructure Upgrade')
proj_4 = find_project_id_by_name('R&D Initiative Alpha')
proj_5 = find_project_id_by_name('Staff Training Program')
proj_6 = find_project_id_by_name('System Automation')
proj_7 = find_project_id_by_name('Global Expansion Pilot')
proj_10 = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in project_ids:
        print(f'x[{i}] {x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')