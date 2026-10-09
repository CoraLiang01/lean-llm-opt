import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    df[col] = df[col].str.strip().astype(int)
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 in project_ids and 7 in project_ids:
    m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
else:
    raise ValueError('Project 4 or 7 not found in project IDs.')
if 6 in project_ids and 1 in project_ids:
    m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
else:
    raise ValueError('Project 6 or 1 not found in project IDs.')
if 10 in project_ids and 5 in project_ids:
    m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
else:
    raise ValueError('Project 10 or 5 not found in project IDs.')
m.optimize()