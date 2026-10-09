import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
project_name = dict(zip(df['Project ID'], df['Project Name']))
required_ids = [1, 4, 5, 6, 7, 10]
missing_ids = [pid for pid in required_ids if pid not in project_ids]
if missing_ids:
    raise ValueError(f'Missing required Project IDs in CSV: {missing_ids}')
m = Model('project_selection')
x_vars = m.addVars(project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(quicksum((npv[i] * x_vars[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()