import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.casefold().str.strip()
project_ids = df['Project ID'].tolist()
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
name_to_id = dict(zip(df['Project Name_norm'], df['Project ID']))

def get_project_id_by_name(target_name):
    norm = target_name.casefold().strip()
    if norm not in name_to_id:
        raise ValueError(f"Project '{target_name}' not found in project.csv")
    return int(name_to_id[norm])
proj_4_id = 4
proj_7_id = 7
proj_6_id = 6
proj_1_id = 1
proj_10_id = 10
proj_5_id = 5
required_ids = [proj_1_id, proj_4_id, proj_5_id, proj_6_id, proj_7_id, proj_10_id]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv')

def solve_project_selection(project_ids, npv, capital):
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_excl_4_7')
    m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_1')
    m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_project_selection(project_ids, npv, capital)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')