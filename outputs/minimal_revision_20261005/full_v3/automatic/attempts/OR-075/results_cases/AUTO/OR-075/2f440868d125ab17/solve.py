import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise ValueError('Missing required columns in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got %d' % len(set(project_ids)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
for pid in project_ids:
    if pid not in capital or pid not in npv:
        raise ValueError(f'Missing capital or NPV for Project ID {pid}')
name_to_id = {n.casefold().strip(): i for (i, n) in zip(df['Project ID'], df['Project Name'])}

def get_pid_by_name(name):
    key = name.casefold().strip()
    if key not in name_to_id:
        raise ValueError(f"Project name '{name}' not found in project.csv")
    return int(name_to_id[key])
pid_1 = get_pid_by_name('Infrastructure Upgrade')
pid_4 = get_pid_by_name('R&D Initiative Alpha')
pid_5 = get_pid_by_name('Staff Training Program')
pid_6 = get_pid_by_name('System Automation')
pid_7 = get_pid_by_name('Global Expansion Pilot')
pid_10 = get_pid_by_name('Customer Experience Platform')

def solve_project_selection(project_ids, capital, npv, pid_1, pid_4, pid_5, pid_6, pid_7, pid_10):
    m = gp.Model('ProjectSelection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[pid_4] + x[pid_7] <= 1, name='mutual_4_7')
    m.addConstr(x[pid_6] <= x[pid_1], name='prereq_6_1')
    m.addConstr(x[pid_10] <= x[pid_5], name='contingent_10_5')
    m.optimize()
    return m
m = solve_project_selection(project_ids, capital, npv, pid_1, pid_4, pid_5, pid_6, pid_7, pid_10)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')