import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Required columns 'NPV (k$)' or 'Capital (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
if 'Project Name' not in df.columns:
    raise KeyError("Required column 'Project Name' not found in project.csv")
name_to_id = {}
for (idx, row) in df.iterrows():
    pname = row['Project Name']
    pid = int(row['Project ID'])
    norm_name = pname.strip().casefold()
    name_to_id[norm_name] = pid

def find_project_id_by_name(target_name):
    norm_target = target_name.strip().casefold()
    if norm_target not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return name_to_id[norm_target]
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
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'].astype(int) == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital_dict[i]} k$, NPV: {npv_dict[i]} k$)')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')