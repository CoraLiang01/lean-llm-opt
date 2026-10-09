import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise ValueError('Missing required columns in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Project ID {pid} required for constraints is missing from project.csv')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_1')
m.addConstr(x[10] <= x[5], name='contingent_10_5')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in project_ids:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')