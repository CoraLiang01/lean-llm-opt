import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
colmap = {norm_col(c): c for c in df.columns}
id_col = colmap['project id']
npv_col = colmap['npv (k$)']
capital_col = colmap['capital (k$)']
name_col = colmap['project name']
df[id_col] = df[id_col].astype(int)
df[npv_col] = df[npv_col].astype(int)
df[capital_col] = df[capital_col].astype(int)
project_ids = df[id_col].tolist()
npv = dict(zip(df[id_col], df[npv_col]))
capital = dict(zip(df[id_col], df[capital_col]))
df['_norm_name'] = df[name_col].str.strip().str.casefold()
name_to_id = dict(zip(df['_norm_name'], df[id_col]))

def get_id_by_name(target_name):
    norm = target_name.strip().casefold()
    if norm not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in CSV.")
    return name_to_id[norm]
proj_4_id = 4
proj_7_id = 7
proj_6_id = 6
proj_1_id = 1
proj_10_id = 10
proj_5_id = 5
for pid in [proj_4_id, proj_7_id, proj_6_id, proj_1_id, proj_10_id, proj_5_id]:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv.')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj_4_id] + y_vars[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(y_vars[proj_6_id] <= y_vars[proj_1_id], name='prereq_6_1')
m.addConstr(y_vars[proj_10_id] <= y_vars[proj_5_id], name='contingent_10_5')
m.optimize()