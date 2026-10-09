import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name'] = df['Project Name'].str.strip()
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
name_dict = dict(zip(df['Project ID'], df['Project Name']))

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {norm(name): pid for (pid, name) in name_dict.items()}
required_projects = [(4, 'R&D Initiative Alpha'), (7, 'Global Expansion Pilot'), (6, 'System Automation'), (1, 'Infrastructure Upgrade'), (10, 'Customer Experience Platform'), (5, 'Staff Training Program')]
for (pid, pname) in required_projects:
    if pid not in project_ids:
        raise ValueError(f'Project ID {pid} ({pname}) not found in project.csv')
    if norm(pname) not in name_to_id or name_to_id[norm(pname)] != pid:
        raise ValueError(f"Project name '{pname}' does not match Project ID {pid} in project.csv")
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            print(f'  Project ID {i:3d}: {name_dict[i]} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')