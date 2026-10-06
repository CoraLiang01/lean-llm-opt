import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
if len(project_ids) != 110 or set(project_ids) != set(range(1, 111)):
    raise ValueError('Project ID set in CSV does not match expected 1..110.')
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
for i in range(1, 111):
    if i not in npv or i not in capital:
        raise ValueError(f'Missing NPV or Capital for Project ID {i}')
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
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'Project {i:3d}: {pname:35s} | Capital: {capital[i]:6.0f} k$ | NPV: {npv[i]:6.0f} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital:.0f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')