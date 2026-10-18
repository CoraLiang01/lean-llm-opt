import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].str.strip()
project_ids = df['Project ID'].astype(int).tolist()
capital_dict = {}
npv_dict = {}
for idx, row in df.iterrows():
    pid = int(row['Project ID'])
    try:
        capital = float(row['Capital (k$)'])
        npv = float(row['NPV (k$)'])
    except Exception as e:
        raise ValueError(f'Non-numeric value in Capital or NPV for Project ID {pid}: {e}')
    capital_dict[pid] = capital
    npv_dict[pid] = npv
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[pid] * y_vars[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[pid] * y_vars[pid] for pid in project_ids)) <= 1000, name='budget')
if 4 not in y_vars or 7 not in y_vars:
    raise KeyError('Project 4 or 7 not found in project.csv')
m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
if 1 not in y_vars or 6 not in y_vars:
    raise KeyError('Project 1 or 6 not found in project.csv')
m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
if 5 not in y_vars or 10 not in y_vars:
    raise KeyError('Project 5 or 10 not found in project.csv')
m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
m.optimize()