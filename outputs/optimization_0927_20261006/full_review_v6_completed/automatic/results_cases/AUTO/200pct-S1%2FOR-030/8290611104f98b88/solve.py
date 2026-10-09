import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Required columns 'NPV (k$)' or 'Capital (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
if 'Project Name' not in df.columns:
    raise KeyError("Required column 'Project Name' not found in project.csv")

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
special_projects = {}
for pname in ['R&D Initiative Alpha', 'Global Expansion Pilot', 'System Automation', 'Infrastructure Upgrade', 'Customer Experience Platform', 'Staff Training Program']:
    key = norm(pname)
    if key not in name_to_id:
        raise ValueError(f"Project '{pname}' not found in project.csv")
    special_projects[pname] = name_to_id[key]
proj_4 = special_projects['R&D Initiative Alpha']
proj_7 = special_projects['Global Expansion Pilot']
proj_6 = special_projects['System Automation']
proj_1 = special_projects['Infrastructure Upgrade']
proj_10 = special_projects['Customer Experience Platform']
proj_5 = special_projects['Staff Training Program']
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj_4] + y_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[proj_6] <= y_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(y_vars[proj_10] <= y_vars[proj_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i:3d}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')