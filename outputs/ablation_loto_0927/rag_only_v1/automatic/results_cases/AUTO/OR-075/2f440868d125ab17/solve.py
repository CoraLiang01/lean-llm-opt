import pandas as pd
import numpy as np
from gurobipy import Model, GRB
project_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
if project_df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = project_df['Project ID'].astype(int).tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got {}'.format(len(set(project_ids))))
npv_dict = dict(zip(project_df['Project ID'].astype(int), project_df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(project_df['Project ID'].astype(int), project_df['Capital (k$)'].astype(int)))
name_to_id = {n.strip().casefold(): i for (i, n) in zip(project_df['Project ID'], project_df['Project Name'])}
required_projects = {'R&D Initiative Alpha': None, 'Global Expansion Pilot': None, 'System Automation': None, 'Infrastructure Upgrade': None, 'Customer Experience Platform': None, 'Staff Training Program': None}
for pname in required_projects:
    key = pname.strip().casefold()
    if key not in name_to_id:
        raise ValueError(f"Project '{pname}' not found in project.csv")
    required_projects[pname] = int(name_to_id[key])
proj_4 = required_projects['R&D Initiative Alpha']
proj_7 = required_projects['Global Expansion Pilot']
proj_6 = required_projects['System Automation']
proj_1 = required_projects['Infrastructure Upgrade']
proj_10 = required_projects['Customer Experience Platform']
proj_5 = required_projects['Staff Training Program']
m = Model('project_selection')
x = m.addVars(project_ids, vtype=GRB.BINARY, name='')
m.setObjective(sum((x[i] * npv_dict[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(sum((x[i] * capital_dict[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4] + x[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6] <= x[proj_1], name='prereq_6_1')
m.addConstr(x[proj_10] <= x[proj_5], name='contingent_10_5')
m.optimize()