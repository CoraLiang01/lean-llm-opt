import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)}')
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
name_to_id = {re.sub('\\s+', ' ', n.strip()).casefold(): i for i, n in zip(df['Project ID'], df['Project Name'])}

def get_pid_by_name(target_name):
    key = re.sub('\\s+', ' ', target_name.strip()).casefold()
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return int(name_to_id[key])
pid_1 = get_pid_by_name('Infrastructure Upgrade')
pid_4 = get_pid_by_name('R&D Initiative Alpha')
pid_5 = get_pid_by_name('Staff Training Program')
pid_6 = get_pid_by_name('System Automation')
pid_7 = get_pid_by_name('Global Expansion Pilot')
pid_10 = get_pid_by_name('Customer Experience Platform')
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
    total_capital = sum((capital[i] * x[i].X for i in project_ids))
    print(f'Total capital used: {total_capital:.2f} k$')
    print('\nSelected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project {i}: {pname} | Capital: {capital[i]} k$ | NPV: {npv[i]} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')