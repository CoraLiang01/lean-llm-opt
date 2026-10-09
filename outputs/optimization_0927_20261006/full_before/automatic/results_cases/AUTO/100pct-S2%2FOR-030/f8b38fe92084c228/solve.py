import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column 'Project ID' in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got %d' % len(set(project_ids)))
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
name_dict = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

def find_project_id_by_name(target_name):
    matches = df[df['Project Name'].str.casefold().str.strip() == target_name.casefold().strip()]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found with name '{target_name}'")
    return int(matches.iloc[0]['Project ID'])
proj_4 = 4
proj_7 = 7
proj_6 = 6
proj_1 = 1
proj_10 = 10
proj_5 = 5
assert name_dict[proj_4].casefold().strip() == 'r&d initiative alpha'
assert name_dict[proj_7].casefold().strip() == 'global expansion pilot'
assert name_dict[proj_6].casefold().strip() == 'system automation'
assert name_dict[proj_1].casefold().strip() == 'infrastructure upgrade'
assert name_dict[proj_10].casefold().strip() == 'customer experience platform'
assert name_dict[proj_5].casefold().strip() == 'staff training program'
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_1')
m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    selected = [i for i in project_ids if x[i].X > 0.5]
    total_npv = sum((npv_dict[i] for i in selected))
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Optimal total NPV: {total_npv:.2f} k$')
    print(f'Total capital used: {total_capital:.2f} k$ (Budget: 1000 k$)')
    print(f'Number of projects selected: {len(selected)}')
    print('\nSelected Projects:')
    for i in selected:
        print(f'  Project ID {i:3d}: {name_dict[i]} | Capital: {capital_dict[i]:.0f} k$ | NPV: {npv_dict[i]:.0f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')