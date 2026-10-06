import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
df.columns = [c.strip() for c in df.columns]
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
name_to_id = {n.casefold().strip(): pid for pid, n in zip(df['Project ID'], df['Project Name'])}

def get_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise KeyError(f"Project name '{target_name}' not found in project.csv")
    return int(name_to_id[key])
proj_4 = 4
proj_7 = 7
proj_6 = 6
proj_1 = 1
proj_10 = 10
proj_5 = 5
for pid in [proj_1, proj_4, proj_5, proj_6, proj_7, proj_10]:
    if pid not in project_ids:
        raise KeyError(f'Project ID {pid} required by constraints not found in project.csv')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_1')
m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_5')
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
    print(f'\nTotal capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')