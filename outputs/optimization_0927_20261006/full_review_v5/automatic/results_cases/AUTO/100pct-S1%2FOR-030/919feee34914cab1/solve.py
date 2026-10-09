import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)'}.issubset(df.columns):
    raise KeyError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(float)
df['NPV (k$)'] = df['NPV (k$)'].astype(float)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def normalize_name(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {normalize_name(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}

def get_project_id_by_name(target_name):
    norm = normalize_name(target_name)
    if norm not in name_to_id:
        raise KeyError(f"Project '{target_name}' not found in project.csv")
    return name_to_id[norm]
proj_1 = get_project_id_by_name('Infrastructure Upgrade')
proj_4 = get_project_id_by_name('R&D Initiative Alpha')
proj_5 = get_project_id_by_name('Staff Training Program')
proj_6 = get_project_id_by_name('System Automation')
proj_7 = get_project_id_by_name('Global Expansion Pilot')
proj_10 = get_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[proj_4] + y_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[proj_6] <= y_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(y_vars[proj_10] <= y_vars[proj_5], name='contingent_10_requires_5')
m.optimize()