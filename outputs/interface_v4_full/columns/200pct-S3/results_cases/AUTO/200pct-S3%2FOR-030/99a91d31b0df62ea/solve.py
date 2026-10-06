import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column 'Project ID' in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)} in project.csv')
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Missing required columns 'NPV (k$)' or 'Capital (k$)' in project.csv")
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[4] + x[7] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x[6] <= x[1], name='prereq_6_requires_1')
m.addConstr(x[10] <= x[5], name='contingent_10_requires_5')
m.optimize()