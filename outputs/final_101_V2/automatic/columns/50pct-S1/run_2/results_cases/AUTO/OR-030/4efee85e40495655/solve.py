import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
if df['Project ID'].duplicated().any():
    raise ValueError('Duplicate Project IDs found in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

def find_project_id_by_name(target_name):
    matches = df[df['Project Name'].str.casefold().str.strip() == target_name.casefold().strip()]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found with name '{target_name}'")
    return int(matches.iloc[0]['Project ID'])
proj_4_id = 4
proj_7_id = 7
proj_6_id = 6
proj_1_id = 1
proj_10_id = 10
proj_5_id = 5
assert proj_name[proj_4_id].casefold().strip() == 'r&d initiative alpha'.casefold().strip()
assert proj_name[proj_7_id].casefold().strip() == 'global expansion pilot'.casefold().strip()
assert proj_name[proj_6_id].casefold().strip() == 'system automation'.casefold().strip()
assert proj_name[proj_1_id].casefold().strip() == 'infrastructure upgrade'.casefold().strip()
assert proj_name[proj_10_id].casefold().strip() == 'customer experience platform'.casefold().strip()
assert proj_name[proj_5_id].casefold().strip() == 'staff training program'.casefold().strip()
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    total_capital = 0.0
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project ID {i:3d}: {proj_name[i]} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
            total_capital += capital[i]
    print(f'Total capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')