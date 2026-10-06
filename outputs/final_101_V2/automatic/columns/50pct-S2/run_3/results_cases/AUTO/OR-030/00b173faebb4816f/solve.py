import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
if set(project_ids) != set(range(1, 111)):
    raise ValueError('Project IDs in CSV do not match expected 1..110.')
capital = df.set_index('Project ID')['Capital (k$)'].astype(int).to_dict()
npv = df.set_index('Project ID')['NPV (k$)'].astype(int).to_dict()
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in CSV.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project {i}: {pname} (Capital: {capital[i]} k$, NPV: {npv[i]} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')