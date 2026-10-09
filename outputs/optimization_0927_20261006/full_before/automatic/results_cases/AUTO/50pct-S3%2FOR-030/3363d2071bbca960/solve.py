import gurobipy as gp
import pandas as pd
import numpy as np
import re
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
proj_name = dict(zip(df['Project ID'], df['Project Name']))
required_projects = {1: 'Infrastructure Upgrade', 4: 'R&D Initiative Alpha', 5: 'Staff Training Program', 6: 'System Automation', 7: 'Global Expansion Pilot', 10: 'Customer Experience Platform'}
for (pid, pname) in required_projects.items():
    if pid not in project_ids:
        raise ValueError(f"Required project ID {pid} ('{pname}') not found in project.csv")
    actual_name = proj_name[pid].strip().casefold()
    expected_name = pname.strip().casefold()
    if actual_name != expected_name:
        raise ValueError(f"Project ID {pid} name mismatch: expected '{pname}', found '{proj_name[pid]}'")
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project ID {i:3d}: {proj_name[i]} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital} k$ (Budget: 1000 k$)')
else:
    print(f'No optimal solution found. Status: {m.status}')