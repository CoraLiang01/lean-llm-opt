import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)'}.issubset(df.columns):
    raise KeyError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for _, row in df.iterrows()}
proj_4_id = name_to_id[norm('R&D Initiative Alpha')]
proj_7_id = name_to_id[norm('Global Expansion Pilot')]
proj_6_id = name_to_id[norm('System Automation')]
proj_1_id = name_to_id[norm('Infrastructure Upgrade')]
proj_10_id = name_to_id[norm('Customer Experience Platform')]
proj_5_id = name_to_id[norm('Staff Training Program')]
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project {i}: {pname} (Capital: {capital_dict[i]} k$, NPV: {npv_dict[i]} k$)')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')