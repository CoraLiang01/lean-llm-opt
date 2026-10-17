import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    df[col] = df[col].astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
name_to_id = {row['Project Name'].strip(): row['Project ID'] for _, row in df.iterrows()}
required_projects = {'R&D Initiative Alpha': None, 'Global Expansion Pilot': None, 'System Automation': None, 'Infrastructure Upgrade': None, 'Customer Experience Platform': None, 'Staff Training Program': None}
for pname in required_projects:
    matches = df[df['Project Name'].str.strip().str.casefold() == pname.strip().casefold()]
    if len(matches) != 1:
        raise ValueError(f"Project '{pname}' not found or not unique in project.csv")
    required_projects[pname] = int(matches.iloc[0]['Project ID'])
proj_4 = required_projects['R&D Initiative Alpha']
proj_7 = required_projects['Global Expansion Pilot']
proj_6 = required_projects['System Automation']
proj_1 = required_projects['Infrastructure Upgrade']
proj_10 = required_projects['Customer Experience Platform']
proj_5 = required_projects['Staff Training Program']
m = Model('project_selection')
x_vars = m.addVars(project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_requires_5')
m.optimize()