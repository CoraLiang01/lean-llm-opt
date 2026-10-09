import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(float)
df['NPV (k$)'] = df['NPV (k$)'].astype(float)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
required_projects = {'R&D Initiative Alpha': 4, 'Global Expansion Pilot': 7, 'System Automation': 6, 'Infrastructure Upgrade': 1, 'Customer Experience Platform': 10, 'Staff Training Program': 5}
for (pname, pid) in required_projects.items():
    if pid not in project_ids:
        raise ValueError(f'Project ID {pid} ({pname}) not found in project.csv')
    row = df[df['Project ID'] == pid]
    if row.empty or norm(row.iloc[0]['Project Name']) != norm(pname):
        raise ValueError(f"Project ID {pid} does not match expected name '{pname}' in project.csv")
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='Budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='MutuallyExclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='Prerequisite_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='Contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'Project ID {i}: {pname} | Capital: {capital_dict[i]:.0f} k$ | NPV: {npv_dict[i]:.0f} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')