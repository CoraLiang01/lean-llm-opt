import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
project_ids = df['Project ID'].astype(int).tolist()
capital_dict = {}
npv_dict = {}
for (idx, row) in df.iterrows():
    pid = int(row['Project ID'])
    try:
        capital = int(row['Capital (k$)'])
        npv = int(row['NPV (k$)'])
    except Exception as e:
        raise ValueError(f'Non-numeric value in Capital (k$) or NPV (k$) for Project ID {pid}: {e}')
    capital_dict[pid] = capital
    npv_dict[pid] = npv
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 not in project_ids or 7 not in project_ids:
    raise KeyError('Project 4 or Project 7 not found in Project ID list')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
if 1 not in project_ids or 6 not in project_ids:
    raise KeyError('Project 1 or Project 6 not found in Project ID list')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
if 5 not in project_ids or 10 not in project_ids:
    raise KeyError('Project 5 or Project 10 not found in Project ID list')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
m.optimize()