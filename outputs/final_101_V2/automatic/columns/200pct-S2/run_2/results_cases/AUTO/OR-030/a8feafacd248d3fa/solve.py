import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')

def norm_col(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
col_id = [c for c in df.columns if norm_col(c) == 'project id'][0]
col_name = [c for c in df.columns if norm_col(c) == 'project name'][0]
col_capital = [c for c in df.columns if norm_col(c) == 'capital (k$)'][0]
col_npv = [c for c in df.columns if norm_col(c) == 'npv (k$)'][0]
project_ids = df[col_id].astype(int).tolist()
if len(set(project_ids)) != len(project_ids):
    raise ValueError('Duplicate Project IDs found in project.csv')
id2idx = {pid: idx for idx, pid in enumerate(project_ids)}
npv = df.set_index(col_id)[col_npv].astype(float).to_dict()
capital = df.set_index(col_id)[col_capital].astype(float).to_dict()
proj_name = df.set_index(col_id)[col_name].astype(str).to_dict()

def find_project_id_by_name(target_name):
    norm_target = re.sub('\\s+', ' ', target_name.strip()).casefold()
    for pid, name in proj_name.items():
        if re.sub('\\s+', ' ', name.strip()).casefold() == norm_target:
            return pid
    raise ValueError(f"Project with name '{target_name}' not found in project.csv")
pid_1 = find_project_id_by_name('Infrastructure Upgrade')
pid_4 = find_project_id_by_name('R&D Initiative Alpha')
pid_5 = find_project_id_by_name('Staff Training Program')
pid_6 = find_project_id_by_name('System Automation')
pid_7 = find_project_id_by_name('Global Expansion Pilot')
pid_10 = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[pid_4] + x[pid_7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[pid_6] <= x[pid_1], name='prereq_6_1')
m.addConstr(x[pid_10] <= x[pid_5], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            print(f'  Project ID {i}: {proj_name[i]} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
    total_cap = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_cap:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')