import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Missing required Project IDs in data: {missing_ids}')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_requires_1')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_requires_5')
m.optimize()