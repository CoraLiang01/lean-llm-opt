import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
name_to_id = {}
for (idx, row) in df.iterrows():
    pname = row['Project Name'].casefold().strip()
    pid = row['Project ID']
    name_to_id[pname] = pid

def find_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise KeyError(f"Project '{target_name}' not found in Project Name column.")
    return name_to_id[key]
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_6_id = find_project_id_by_name('System Automation')
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
proj_5_id = find_project_id_by_name('Staff Training Program')

def solve_project_selection(project_ids, capital, npv, proj_4_id, proj_7_id, proj_6_id, proj_1_id, proj_10_id, proj_5_id):
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_4_7')
    m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_1')
    m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_project_selection(project_ids=project_ids, capital=capital, npv=npv, proj_4_id=proj_4_id, proj_7_id=proj_7_id, proj_6_id=proj_6_id, proj_1_id=proj_1_id, proj_10_id=proj_10_id, proj_5_id=proj_5_id)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')