import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

def find_project_id_by_name(target_name):
    norm = lambda s: re.sub('\\s+', ' ', s.strip().casefold())
    matches = [pid for pid, name in proj_name.items() if norm(name) == norm(target_name)]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project '{target_name}' in data (found {len(matches)} matches).")
    return matches[0]
proj_id_1 = find_project_id_by_name('Infrastructure Upgrade')
proj_id_4 = find_project_id_by_name('R&D Initiative Alpha')
proj_id_5 = find_project_id_by_name('Staff Training Program')
proj_id_6 = find_project_id_by_name('System Automation')
proj_id_7 = find_project_id_by_name('Global Expansion Pilot')
proj_id_10 = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_id_4] + x[proj_id_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[proj_id_6] <= x[proj_id_1], name='prereq_6_1')
m.addConstr(x[proj_id_10] <= x[proj_id_5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    total_capital = 0.0
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project ID {i:3d}: {proj_name[i]} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            total_capital += capital[i]
    print(f'Total capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')