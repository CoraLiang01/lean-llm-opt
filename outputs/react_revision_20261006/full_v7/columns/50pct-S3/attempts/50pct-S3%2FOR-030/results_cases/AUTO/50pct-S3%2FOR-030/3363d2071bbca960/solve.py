import gurobipy as gp
import pandas as pd
import numpy as np
import re
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',', dtype=str, keep_default_na=False)
required_columns = ['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
project_names = dict(zip(df['Project ID'], df['Project Name']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(target_name):
    norm_target = re.sub('\\s+', ' ', target_name.strip()).casefold()
    for (pid, name) in project_names.items():
        norm_name = re.sub('\\s+', ' ', name.strip()).casefold()
        if norm_name == norm_target:
            return pid
    raise ValueError(f"Project with name '{target_name}' not found in data.")
pid_4 = find_project_id_by_name('R&D Initiative Alpha')
pid_7 = find_project_id_by_name('Global Expansion Pilot')
pid_6 = find_project_id_by_name('System Automation')
pid_1 = find_project_id_by_name('Infrastructure Upgrade')
pid_10 = find_project_id_by_name('Customer Experience Platform')
pid_5 = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x_vars[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[pid_4] + x_vars[pid_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[pid_6] <= x_vars[pid_1], name='prereq_6_1')
m.addConstr(x_vars[pid_10] <= x_vars[pid_5], name='contingent_10_5')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in project_ids:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')