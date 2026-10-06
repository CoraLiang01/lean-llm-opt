import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects}')
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))

def find_project_id_by_name(name):
    norm = lambda s: re.sub('\\s+', ' ', s.strip().casefold())
    matches = [pid for pid, pname in proj_name.items() if norm(pname) == norm(name)]
    if not matches:
        raise ValueError(f"Project '{name}' not found in project.csv")
    return matches[0]
id_1 = find_project_id_by_name('Infrastructure Upgrade')
id_4 = find_project_id_by_name('R&D Initiative Alpha')
id_5 = find_project_id_by_name('Staff Training Program')
id_6 = find_project_id_by_name('System Automation')
id_7 = find_project_id_by_name('Global Expansion Pilot')
id_10 = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
y = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * y[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * y[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y[id_4] + y[id_7] <= 1, name='mutual_excl_4_7')
m.addConstr(y[id_6] <= y[id_1], name='prereq_6_1')
m.addConstr(y[id_10] <= y[id_5], name='contingent_10_5')
m.optimize()