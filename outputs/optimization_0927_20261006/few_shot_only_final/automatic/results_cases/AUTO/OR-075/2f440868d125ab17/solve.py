import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
project_ids = df['Project ID'].tolist()
npv_dict = {}
capital_dict = {}
for (idx, row) in df.iterrows():
    pid = str(row['Project ID'])
    try:
        npv = float(row['NPV (k$)'])
        capital = float(row['Capital (k$)'])
    except Exception as e:
        raise ValueError(f'Non-numeric NPV or Capital for Project ID {pid}: {e}')
    npv_dict[pid] = npv
    capital_dict[pid] = capital
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[pid] * x_vars[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[pid] * x_vars[pid] for pid in project_ids)) <= 1000, name='budget')
if '4' not in project_ids or '7' not in project_ids:
    raise KeyError("Project ID '4' or '7' not found in project.csv")
m.addConstr(x_vars['4'] + x_vars['7'] <= 1, name='mutual_4_7')
if '6' not in project_ids or '1' not in project_ids:
    raise KeyError("Project ID '6' or '1' not found in project.csv")
m.addConstr(x_vars['6'] <= x_vars['1'], name='prereq_6_1')
if '10' not in project_ids or '5' not in project_ids:
    raise KeyError("Project ID '10' or '5' not found in project.csv")
m.addConstr(x_vars['10'] <= x_vars['5'], name='contingent_10_5')
m.optimize()