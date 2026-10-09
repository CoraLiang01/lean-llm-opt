import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Missing required Project IDs in CSV: {missing_ids}')
m = gp.Model('ProjectSelection')
y = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y[4] + y[7] <= 1, name='mutual_excl_4_7')
m.addConstr(y[6] <= y[1], name='prereq_6_1')
m.addConstr(y[10] <= y[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    selected = []
    for i in project_ids:
        if y[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i:3d}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            selected.append(i)
    total_cap = sum((capital[i] for i in selected))
    print(f'\nTotal capital used: {total_cap:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')