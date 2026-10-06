import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
if len(npv) != 110 or len(capital) != 110:
    raise ValueError('Project count mismatch: expected 110 projects with NPV and Capital data.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')

def find_project_id_by_name(name):
    matches = df[df['Project Name'].str.casefold().str.strip() == name.casefold().strip()]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
proj4_id = find_project_id_by_name('R&D Initiative Alpha')
proj7_id = find_project_id_by_name('Global Expansion Pilot')
m.addConstr(x[proj4_id] + x[proj7_id] <= 1, name='mutual_exclusive_4_7')
proj6_id = find_project_id_by_name('System Automation')
proj1_id = find_project_id_by_name('Infrastructure Upgrade')
m.addConstr(x[proj6_id] <= x[proj1_id], name='prereq_6_requires_1')
proj10_id = find_project_id_by_name('Customer Experience Platform')
proj5_id = find_project_id_by_name('Staff Training Program')
m.addConstr(x[proj10_id] <= x[proj5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital[i]:.0f} k$, NPV: {npv[i]:.0f} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')