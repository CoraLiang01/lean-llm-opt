import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
project_ids = df['Project ID'].astype(int).tolist()
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 in project_ids and 7 in project_ids:
    m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_excl_4_7')
else:
    raise ValueError('Project 4 and/or Project 7 not found in project IDs.')
if 1 in project_ids and 6 in project_ids:
    m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
else:
    raise ValueError('Project 1 and/or Project 6 not found in project IDs.')
if 5 in project_ids and 10 in project_ids:
    m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
else:
    raise ValueError('Project 5 and/or Project 10 not found in project IDs.')
m.optimize()