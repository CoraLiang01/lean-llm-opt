import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
proj_name = dict(zip(df['Project ID'], df['Project Name']))

def find_project_id_by_name(target_name):
    norm_target = re.sub('\\s+', ' ', target_name.strip().casefold())
    for (pid, name) in proj_name.items():
        norm_name = re.sub('\\s+', ' ', name.strip().casefold())
        if norm_name == norm_target:
            return pid
    raise ValueError(f"Project with name '{target_name}' not found.")
assert find_project_id_by_name('Infrastructure Upgrade') == 1
assert find_project_id_by_name('R&D Initiative Alpha') == 4
assert find_project_id_by_name('Staff Training Program') == 5
assert find_project_id_by_name('System Automation') == 6
assert find_project_id_by_name('Global Expansion Pilot') == 7
assert find_project_id_by_name('Customer Experience Platform') == 10
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
m.optimize()