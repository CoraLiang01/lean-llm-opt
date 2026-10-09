import gurobipy as gp
import pandas as pd
import numpy as np
import re
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))

def find_project_id_by_name(name):
    matches = df[df['Project Name'].str.casefold().str.strip() == name.casefold().strip()]
    if len(matches) != 1:
        raise ValueError(f"Project name '{name}' matched {len(matches)} rows, expected exactly 1.")
    return int(matches.iloc[0]['Project ID'])
proj_id_1 = find_project_id_by_name('Infrastructure Upgrade')
proj_id_4 = find_project_id_by_name('R&D Initiative Alpha')
proj_id_5 = find_project_id_by_name('Staff Training Program')
proj_id_6 = find_project_id_by_name('System Automation')
proj_id_7 = find_project_id_by_name('Global Expansion Pilot')
proj_id_10 = find_project_id_by_name('Customer Experience Platform')
for pid in [proj_id_1, proj_id_4, proj_id_5, proj_id_6, proj_id_7, proj_id_10]:
    if pid not in project_ids:
        raise KeyError(f'Required Project ID {pid} not found in project_ids.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_id_4] + x[proj_id_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_id_6] <= x[proj_id_1], name='prereq_6_requires_1')
m.addConstr(x[proj_id_10] <= x[proj_id_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'Project ID {i:3d}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital[i] for i in selected))
    print(f'Total capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')