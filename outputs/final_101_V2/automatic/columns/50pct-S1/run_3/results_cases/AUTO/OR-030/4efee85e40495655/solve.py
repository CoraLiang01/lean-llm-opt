import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
required_projects = {1: 'Infrastructure Upgrade', 4: 'R&D Initiative Alpha', 5: 'Staff Training Program', 6: 'System Automation', 7: 'Global Expansion Pilot', 10: 'Customer Experience Platform'}
for pid, pname in required_projects.items():
    if pid not in project_ids:
        raise ValueError(f"Required project ID {pid} ('{pname}') not found in project.csv")
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f} (Total NPV in k$)')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i:3d}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            selected.append(i)
    print(f'\nTotal capital used: {sum((capital[i] for i in selected)):.0f} / 1000 k$')
    print(f'Number of projects selected: {len(selected)} / {len(project_ids)}')
else:
    print(f'No optimal solution found. Status: {m.status}')