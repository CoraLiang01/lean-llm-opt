import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
    df[col] = df[col].str.strip().astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def norm(s):
    return s.strip().casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
project_1_id = name_to_id[norm('Infrastructure Upgrade')]
project_4_id = name_to_id[norm('R&D Initiative Alpha')]
project_5_id = name_to_id[norm('Staff Training Program')]
project_6_id = name_to_id[norm('System Automation')]
project_7_id = name_to_id[norm('Global Expansion Pilot')]
project_10_id = name_to_id[norm('Customer Experience Platform')]
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[project_4_id] + x_vars[project_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[project_6_id] <= x_vars[project_1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[project_10_id] <= x_vars[project_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            row = df[df['Project ID'] == i].iloc[0]
            print(f"  Project ID {i}: {row['Project Name']} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$")
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')