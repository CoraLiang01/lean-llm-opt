import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)}')
for i in project_ids:
    if i not in npv or i not in capital:
        raise ValueError(f'Missing NPV or Capital for Project ID {i}')

def find_project_id_by_name(target_name):
    norm = lambda s: re.sub('\\s+', ' ', s.strip().casefold())
    target = norm(target_name)
    for pid, name in proj_name.items():
        if norm(name) == target:
            return pid
    raise ValueError(f"Project with name '{target_name}' not found.")
proj_4 = find_project_id_by_name('R&D Initiative Alpha')
proj_7 = find_project_id_by_name('Global Expansion Pilot')
proj_6 = find_project_id_by_name('System Automation')
proj_1 = find_project_id_by_name('Infrastructure Upgrade')
proj_10 = find_project_id_by_name('Customer Experience Platform')
proj_5 = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_1')
m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project ID {i}: {proj_name[i]} (Capital: {capital[i]:.0f} k$, NPV: {npv[i]:.0f} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')