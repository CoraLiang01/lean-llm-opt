import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
df = df.set_index('Project ID', drop=False)
for col in ['Capital (k$)', 'NPV (k$)']:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
    df[col] = df[col].astype(int)
project_ids = df.index.tolist()
npv = df['NPV (k$)'].to_dict()
capital = df['Capital (k$)'].to_dict()

def norm(s):
    return s.strip().casefold()
name_to_id = {norm(row['Project Name']): pid for (pid, row) in df.iterrows()}

def find_project_id_by_name(target_name):
    n_target = norm(target_name)
    matches = [pid for (pid, row) in df.iterrows() if norm(row['Project Name']) == n_target]
    if len(matches) == 0:
        raise ValueError(f"Project with name '{target_name}' not found in project.csv")
    if len(matches) > 1:
        raise ValueError(f"Multiple projects found with name '{target_name}'")
    return matches[0]
proj_4_id = 4
if proj_4_id not in project_ids or norm(df.loc[proj_4_id]['Project Name']) != norm('R&D Initiative Alpha'):
    proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_7_id = 7
if proj_7_id not in project_ids or norm(df.loc[proj_7_id]['Project Name']) != norm('Global Expansion Pilot'):
    proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_6_id = 6
if proj_6_id not in project_ids or norm(df.loc[proj_6_id]['Project Name']) != norm('System Automation'):
    proj_6_id = find_project_id_by_name('System Automation')
proj_1_id = 1
if proj_1_id not in project_ids or norm(df.loc[proj_1_id]['Project Name']) != norm('Infrastructure Upgrade'):
    proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_10_id = 10
if proj_10_id not in project_ids or norm(df.loc[proj_10_id]['Project Name']) != norm('Customer Experience Platform'):
    proj_10_id = find_project_id_by_name('Customer Experience Platform')
proj_5_id = 5
if proj_5_id not in project_ids or norm(df.loc[proj_5_id]['Project Name']) != norm('Staff Training Program'):
    proj_5_id = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    total_capital = 0
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[i]['Project Name']
            print(f'  Project {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
            total_capital += capital[i]
    print(f'Total Capital Used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')