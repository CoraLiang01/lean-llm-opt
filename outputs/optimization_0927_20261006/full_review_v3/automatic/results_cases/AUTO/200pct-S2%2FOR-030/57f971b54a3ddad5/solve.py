import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Required column 'Project ID' not found in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Required columns 'NPV (k$)' or 'Capital (k$)' not found in project.csv")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
special_ids = [1, 4, 5, 6, 7, 10]
for pid in special_ids:
    if pid not in project_ids:
        raise ValueError(f'Project ID {pid} required by constraints not found in project.csv')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()