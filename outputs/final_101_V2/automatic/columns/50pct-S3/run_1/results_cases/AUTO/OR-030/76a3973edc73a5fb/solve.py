import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df.shape[0] != 110:
    raise ValueError(f'Expected 110 projects, found {df.shape[0]} rows in project.csv.')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

def find_project_id_by_name(target_name):
    norm = lambda s: re.sub('\\s+', ' ', s.strip().casefold())
    target_norm = norm(target_name)
    matches = [pid for pid, name in proj_name.items() if norm(name) == target_norm]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv.")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found for name '{target_name}': {matches}")
    return matches[0]
pid_4 = find_project_id_by_name('R&D Initiative Alpha')
pid_7 = find_project_id_by_name('Global Expansion Pilot')
if pid_4 != 4 or pid_7 != 7:
    raise ValueError(f'Expected Project IDs 4 and 7 for the named projects, got {pid_4} and {pid_7}')
pid_6 = find_project_id_by_name('System Automation')
pid_1 = find_project_id_by_name('Infrastructure Upgrade')
if pid_6 != 6 or pid_1 != 1:
    raise ValueError(f'Expected Project IDs 6 and 1 for the named projects, got {pid_6} and {pid_1}')
pid_10 = find_project_id_by_name('Customer Experience Platform')
pid_5 = find_project_id_by_name('Staff Training Program')
if pid_10 != 10 or pid_5 != 5:
    raise ValueError(f'Expected Project IDs 10 and 5 for the named projects, got {pid_10} and {pid_5}')
m = gp.Model('ProjectSelection')
y = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((y[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((y[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y[4] + y[7] <= 1, name='mutual_excl_4_7')
m.addConstr(y[6] <= y[1], name='prereq_6_1')
m.addConstr(y[10] <= y[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total NPV in k$)')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if y[i].X > 0.5:
            selected.append((i, proj_name[i], capital[i], npv[i]))
    selected.sort()
    total_cap = sum((row[2] for row in selected))
    for i, name, cap, n in selected:
        print(f'  Project {i:3d}: {name:35s} | Capital: {cap:6.1f} k$ | NPV: {n:6.1f} k$')
    print('-------------------------------------------------')
    print(f'Total Capital Used: {total_cap:.1f} / 1000.0 k$')
    print(f'Number of Projects Selected: {len(selected)}')
else:
    print(f'No optimal solution found. Status: {m.status}')