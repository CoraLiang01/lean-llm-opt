import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if not set(['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']).issubset(df.columns):
    raise ValueError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
project_name = dict(zip(df['Project ID'], df['Project Name']))
special_projects = {4: 'R&D Initiative Alpha', 7: 'Global Expansion Pilot', 6: 'System Automation', 1: 'Infrastructure Upgrade', 10: 'Customer Experience Platform', 5: 'Staff Training Program'}
for (pid, pname) in special_projects.items():
    if pid not in project_ids:
        raise ValueError(f'Special project ID {pid} not found in project.csv')
    if project_name[pid].strip().casefold() != pname.strip().casefold():
        raise ValueError(f"Project ID {pid} name mismatch: expected '{pname}', found '{project_name[pid]}'")
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x_vars[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x_vars[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in project_ids:
        print(f'x[{i}] {x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')