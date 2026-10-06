import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
name_to_id = {}
for idx, row in df.iterrows():
    pname = row['Project Name']
    pid = int(row['Project ID'])
    key = re.sub('\\s+', ' ', pname.strip().casefold())
    name_to_id[key] = pid

def get_pid_by_name(target_name):
    key = re.sub('\\s+', ' ', target_name.strip().casefold())
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in CSV.")
    return name_to_id[key]
pid_1 = get_pid_by_name('Infrastructure Upgrade')
pid_4 = get_pid_by_name('R&D Initiative Alpha')
pid_5 = get_pid_by_name('Staff Training Program')
pid_6 = get_pid_by_name('System Automation')
pid_7 = get_pid_by_name('Global Expansion Pilot')
pid_10 = get_pid_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='x')
m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='Budget')
m.addConstr(x[pid_4] + x[pid_7] <= 1, name='MutualExcl_4_7')
m.addConstr(x[pid_6] <= x[pid_1], name='Prereq_6_requires_1')
m.addConstr(x[pid_10] <= x[pid_5], name='Contingent_10_requires_5')
m.optimize()