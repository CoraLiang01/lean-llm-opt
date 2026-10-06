import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
required_projects = {1: 'Infrastructure Upgrade', 4: 'R&D Initiative Alpha', 5: 'Staff Training Program', 6: 'System Automation', 7: 'Global Expansion Pilot', 10: 'Customer Experience Platform'}
for pid, pname in required_projects.items():
    if pid not in project_ids:
        raise ValueError(f"Required project ID {pid} ('{pname}') not found in project.csv")
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_requires_1')
m.addConstr(x[10] <= x[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')