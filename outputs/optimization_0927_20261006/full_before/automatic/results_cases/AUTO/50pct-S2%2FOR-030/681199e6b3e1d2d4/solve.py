import gurobipy as gp
import pandas as pd
import numpy as np
import re
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))

def find_project_id_by_name(name):
    norm = lambda s: re.sub('\\s+', ' ', s.strip().casefold())
    matches = df[df['Project Name'].apply(lambda x: norm(x) == norm(name))]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_5_id = find_project_id_by_name('Staff Training Program')
proj_6_id = find_project_id_by_name('System Automation')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    selected = []
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital[i]} k$, NPV: {npv[i]} k$)')
            selected.append(i)
    total_capital = sum((capital[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')