import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
project_ids = sorted(df['Project ID'].unique())
if len(project_ids) != 110 or min(project_ids) != 1 or max(project_ids) != 110:
    raise ValueError('Project IDs in CSV do not match expected range 1..110.')
try:
    capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
    npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Capital/NPV columns to int: {e}')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in project_ids or 7 not in project_ids:
    raise ValueError('Project 4 or 7 not found in Project IDs.')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise ValueError('Project 1 or 6 not found in Project IDs.')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise ValueError('Project 5 or 10 not found in Project IDs.')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
m.optimize()