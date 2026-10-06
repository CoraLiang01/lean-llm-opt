import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
if set(project_ids) != set(npv_dict.keys()) or set(project_ids) != set(capital_dict.keys()):
    raise ValueError('Mismatch in project IDs between index set and NPV/Capital data.')

def get_project_id_by_name(name):
    matches = df[df['Project Name'].str.casefold().str.strip() == name.casefold().strip()]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project '{name}' in data.")
    return int(matches.iloc[0]['Project ID'])
pid_1 = get_project_id_by_name('Infrastructure Upgrade')
pid_4 = get_project_id_by_name('R&D Initiative Alpha')
pid_5 = get_project_id_by_name('Staff Training Program')
pid_6 = get_project_id_by_name('System Automation')
pid_7 = get_project_id_by_name('Global Expansion Pilot')
pid_10 = get_project_id_by_name('Customer Experience Platform')

def solve_project_selection(project_ids, npv_dict, capital_dict, pid_1, pid_4, pid_5, pid_6, pid_7, pid_10):
    m = gp.Model('ProjectSelection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((x[i] * npv_dict[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[i] * capital_dict[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[pid_4] + x[pid_7] <= 1, name='mutual_4_7')
    m.addConstr(x[pid_6] <= x[pid_1], name='prereq_6_1')
    m.addConstr(x[pid_10] <= x[pid_5], name='contingent_10_5')
    m.optimize()
    return m
m = solve_project_selection(project_ids, npv_dict, capital_dict, pid_1, pid_4, pid_5, pid_6, pid_7, pid_10)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')