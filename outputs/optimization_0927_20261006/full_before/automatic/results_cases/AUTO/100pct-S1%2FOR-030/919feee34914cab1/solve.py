import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
name_to_id = {n.casefold().strip(): pid for (n, pid) in zip(df['Project Name'], df['Project ID'])}

def get_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in CSV.")
    return int(name_to_id[key])
proj_4_id = 4
proj_7_id = 7
proj_6_id = 6
proj_1_id = 1
proj_10_id = 10
proj_5_id = 5
for pid in [proj_4_id, proj_7_id, proj_6_id, proj_1_id, proj_10_id, proj_5_id]:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_requires_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    selected = []
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            selected.append(i)
    total_capital = sum((capital[i] for i in selected))
    print(f'\nTotal capital used: {total_capital:.2f} k$ (Budget: 1000 k$)')
else:
    print(f'No optimal solution found. Status: {m.status}')