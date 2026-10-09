import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)', 'Project Name'}.issubset(df.columns):
    raise KeyError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def get_project_id_by_name(name):
    norm_name = name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm_name]
    if len(matches) != 1:
        raise ValueError(f"Project name '{name}' not found uniquely in project.csv")
    return int(matches['Project ID'].iloc[0])
proj_1 = get_project_id_by_name('Infrastructure Upgrade')
proj_4 = get_project_id_by_name('R&D Initiative Alpha')
proj_5 = get_project_id_by_name('Staff Training Program')
proj_6 = get_project_id_by_name('System Automation')
proj_7 = get_project_id_by_name('Global Expansion Pilot')
proj_10 = get_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_requires_5')
m.optimize()