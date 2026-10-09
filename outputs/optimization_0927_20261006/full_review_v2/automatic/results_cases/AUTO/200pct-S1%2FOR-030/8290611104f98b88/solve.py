import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(x):
    return str(x).strip().casefold()
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
project_ids = df['Project ID'].astype(int).tolist()
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
name_map = {norm_str(n): pid for (pid, n) in zip(df['Project ID'].astype(int), df['Project Name'])}

def find_project_id_by_name(target_name):
    key = norm_str(target_name)
    if key not in name_map:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return name_map[key]
pid_4 = find_project_id_by_name('R&D Initiative Alpha')
pid_7 = find_project_id_by_name('Global Expansion Pilot')
pid_6 = find_project_id_by_name('System Automation')
pid_1 = find_project_id_by_name('Infrastructure Upgrade')
pid_10 = find_project_id_by_name('Customer Experience Platform')
pid_5 = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection_NPV_Max')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[pid_4] + y_vars[pid_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[pid_6] <= y_vars[pid_1], name='prereq_6_requires_1')
m.addConstr(y_vars[pid_10] <= y_vars[pid_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'].astype(int) == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')