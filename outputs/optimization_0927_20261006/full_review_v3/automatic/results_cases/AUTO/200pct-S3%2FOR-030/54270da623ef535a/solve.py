import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column: 'Project ID'")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Missing required columns: 'NPV (k$)' and/or 'Capital (k$)'")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in project_ids or 7 not in project_ids:
    raise KeyError('Project IDs 4 and/or 7 not found in project data.')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_exclusive_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise KeyError('Project IDs 1 and/or 6 not found in project data.')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
if 5 not in project_ids or 10 not in project_ids:
    raise KeyError('Project IDs 5 and/or 10 not found in project data.')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()