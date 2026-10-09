import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)', 'Project Name'}.issubset(df.columns):
    raise ValueError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.casefold().str.strip()
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
proj_name = dict(zip(df['Project ID'], df['Project Name']))

def find_project_id_by_name(target_name):
    norm = target_name.casefold().strip()
    matches = df[df['Project Name_norm'] == norm]['Project ID'].tolist()
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found: {matches}")
    return matches[0]
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_5_id = find_project_id_by_name('Staff Training Program')
proj_6_id = find_project_id_by_name('System Automation')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in project_ids:
        print(f'x[{i}] {x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')