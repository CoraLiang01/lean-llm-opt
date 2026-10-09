import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 in y_vars and 7 in y_vars:
    m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
else:
    raise ValueError('Project 4 or Project 7 not found in project IDs.')
if 6 in y_vars and 1 in y_vars:
    m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
else:
    raise ValueError('Project 6 or Project 1 not found in project IDs.')
if 10 in y_vars and 5 in y_vars:
    m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
else:
    raise ValueError('Project 10 or Project 5 not found in project IDs.')
m.optimize()