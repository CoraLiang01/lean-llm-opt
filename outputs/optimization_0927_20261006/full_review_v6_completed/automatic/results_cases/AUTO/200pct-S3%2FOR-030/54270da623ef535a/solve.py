import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Required columns 'NPV (k$)' or 'Capital (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
name_to_id = {}
for (idx, row) in df.iterrows():
    pname = row['Project Name'].strip().casefold()
    pid = row['Project ID']
    name_to_id[pname] = pid

def get_project_id_by_name(target_name):
    key = target_name.strip().casefold()
    if key not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in project.csv")
    return name_to_id[key]
proj4_id = get_project_id_by_name('R&D Initiative Alpha')
proj7_id = get_project_id_by_name('Global Expansion Pilot')
proj6_id = get_project_id_by_name('System Automation')
proj1_id = get_project_id_by_name('Infrastructure Upgrade')
proj10_id = get_project_id_by_name('Customer Experience Platform')
proj5_id = get_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj4_id] + y_vars[proj7_id] <= 1, name='mutually_exclusive_4_7')
m.addConstr(y_vars[proj6_id] <= y_vars[proj1_id], name='prerequisite_6_requires_1')
m.addConstr(y_vars[proj10_id] <= y_vars[proj5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')