import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].duplicated().any():
    raise ValueError('Duplicate Project IDs found in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
required_ids = [1, 4, 5, 6, 7, 10]
missing = [pid for pid in required_ids if pid not in project_ids]
if missing:
    raise ValueError(f'Required Project IDs for constraints not found in data: {missing}')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_requires_1')
m.addConstr(x[10] <= x[5], name='contingent_10_requires_5')
m.optimize()