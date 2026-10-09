import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)'}.issubset(df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].str.strip().astype(int)
df['Capital (k$)'] = df['Capital (k$)'].str.strip().astype(int)
df['NPV (k$)'] = df['NPV (k$)'].str.strip().astype(int)
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Required Project IDs for constraints not found in data: {missing_ids}')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f} (Total NPV in k$)')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')