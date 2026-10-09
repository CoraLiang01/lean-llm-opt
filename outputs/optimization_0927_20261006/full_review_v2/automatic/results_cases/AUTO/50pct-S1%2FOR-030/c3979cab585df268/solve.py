import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))

def find_project_id_by_name(target_name):
    norm_target = target_name.strip().casefold()
    matches = df[df['Project Name'].str.strip().str.casefold() == norm_target]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'")
    return int(matches.iloc[0]['Project ID'])
proj1_id = find_project_id_by_name('Infrastructure Upgrade')
proj4_id = find_project_id_by_name('R&D Initiative Alpha')
proj5_id = find_project_id_by_name('Staff Training Program')
proj6_id = find_project_id_by_name('System Automation')
proj7_id = find_project_id_by_name('Global Expansion Pilot')
proj10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj4_id] + x_vars[proj7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj6_id] <= x_vars[proj1_id], name='prereq_6_1')
m.addConstr(x_vars[proj10_id] <= x_vars[proj5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total NPV in k$)')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i:3d}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    print(f'Total selected: {len(selected)} projects')
    total_cap = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_cap} / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')