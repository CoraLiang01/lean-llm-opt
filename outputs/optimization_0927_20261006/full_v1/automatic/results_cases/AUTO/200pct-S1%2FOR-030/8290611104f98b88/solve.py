import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
project_ids = df['Project ID'].astype(int).tolist()
capital = df.set_index(df['Project ID'].astype(int))['Capital (k$)'].astype(int).to_dict()
npv = df.set_index(df['Project ID'].astype(int))['NPV (k$)'].astype(int).to_dict()
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in project_ids or 7 not in project_ids:
    raise KeyError('Project 4 or 7 not found in project.csv')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise KeyError('Project 1 or 6 not found in project.csv')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise KeyError('Project 5 or 10 not found in project.csv')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
m.optimize()