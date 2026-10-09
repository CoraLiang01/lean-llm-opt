import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if df.shape[0] != 110:
    raise ValueError(f'Expected 110 projects, found {df.shape[0]} in project.csv')
if not np.issubdtype(df['Project ID'].dtype, np.integer):
    df['Project ID'] = df['Project ID'].apply(lambda x: int(x.strip()))
project_ids = df['Project ID'].tolist()
project_id_set = set(project_ids)
try:
    npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].apply(lambda x: float(x.strip()))))
    capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].apply(lambda x: float(x.strip()))))
except Exception as e:
    raise ValueError(f'Error converting NPV or Capital columns to float: {e}')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
for pid in [4, 7]:
    if pid not in project_id_set:
        raise KeyError(f'Project ID {pid} not found in project.csv')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_4_7')
for pid in [1, 6]:
    if pid not in project_id_set:
        raise KeyError(f'Project ID {pid} not found in project.csv')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
for pid in [5, 10]:
    if pid not in project_id_set:
        raise KeyError(f'Project ID {pid} not found in project.csv')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
m.optimize()