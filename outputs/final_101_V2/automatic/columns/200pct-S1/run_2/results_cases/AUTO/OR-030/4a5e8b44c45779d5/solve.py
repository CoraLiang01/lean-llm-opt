import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
name_to_id = {n.casefold().strip(): pid for n, pid in zip(df['Project Name'], df['Project ID'])}

def get_project_id_by_name(target_name):
    key = target_name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in 'Project Name' column.")
    return int(name_to_id[key])
proj_1_id = get_project_id_by_name('Infrastructure Upgrade')
proj_4_id = get_project_id_by_name('R&D Initiative Alpha')
proj_5_id = get_project_id_by_name('Staff Training Program')
proj_6_id = get_project_id_by_name('System Automation')
proj_7_id = get_project_id_by_name('Global Expansion Pilot')
proj_10_id = get_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital[i]:.0f} k$, NPV: {npv[i]:.0f} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')