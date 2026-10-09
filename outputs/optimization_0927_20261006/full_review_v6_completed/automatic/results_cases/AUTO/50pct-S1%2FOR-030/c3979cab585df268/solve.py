import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df = df.set_index('Project ID', drop=False)
project_ids = df.index.tolist()
npv = df['NPV (k$)'].to_dict()
capital = df['Capital (k$)'].to_dict()
name_to_id = {row['Project Name'].strip().casefold(): pid for (pid, row) in df.iterrows()}

def get_project_id_by_name(target_name):
    key = target_name.strip().casefold()
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return name_to_id[key]
proj_4 = get_project_id_by_name('R&D Initiative Alpha')
proj_7 = get_project_id_by_name('Global Expansion Pilot')
proj_6 = get_project_id_by_name('System Automation')
proj_1 = get_project_id_by_name('Infrastructure Upgrade')
proj_10 = get_project_id_by_name('Customer Experience Platform')
proj_5 = get_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_requires_5')
m.optimize()