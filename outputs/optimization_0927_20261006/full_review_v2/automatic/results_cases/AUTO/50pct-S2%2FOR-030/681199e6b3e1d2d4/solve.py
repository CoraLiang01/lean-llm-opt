import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
try:
    npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
    capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting NPV or Capital columns to int: {e}')
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()

def get_project_id_by_name(name):
    norm_name = name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm_name]['Project ID']
    if len(matches) != 1:
        raise ValueError(f"Project name '{name}' matched {len(matches)} projects (expected 1).")
    return int(matches.iloc[0])
proj_4_id = 4
proj_7_id = 7
proj_6_id = 6
proj_1_id = 1
proj_10_id = 10
proj_5_id = 5
assert get_project_id_by_name('R&D Initiative Alpha') == proj_4_id
assert get_project_id_by_name('Global Expansion Pilot') == proj_7_id
assert get_project_id_by_name('System Automation') == proj_6_id
assert get_project_id_by_name('Infrastructure Upgrade') == proj_1_id
assert get_project_id_by_name('Customer Experience Platform') == proj_10_id
assert get_project_id_by_name('Staff Training Program') == proj_5_id
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj_4_id] + y_vars[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[proj_6_id] <= y_vars[proj_1_id], name='prereq_6_requires_1')
m.addConstr(y_vars[proj_10_id] <= y_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')