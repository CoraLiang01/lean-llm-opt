import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].duplicated().any():
    raise ValueError('Duplicate Project IDs found in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='Budget')
if 4 in project_ids and 7 in project_ids:
    m.addConstr(x[4] + x[7] <= 1, name='MutualExcl_4_7')
else:
    raise ValueError('Project 4 or 7 not found in project.csv')
if 6 in project_ids and 1 in project_ids:
    m.addConstr(x[6] <= x[1], name='Prereq_6_1')
else:
    raise ValueError('Project 6 or 1 not found in project.csv')
if 10 in project_ids and 5 in project_ids:
    m.addConstr(x[10] <= x[5], name='Contingent_10_5')
else:
    raise ValueError('Project 10 or 5 not found in project.csv')
m.optimize()