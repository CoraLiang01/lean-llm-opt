import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].str.strip().astype(int)
df['Capital (k$)'] = df['Capital (k$)'].str.strip().astype(int)
df['NPV (k$)'] = df['NPV (k$)'].str.strip().astype(int)
df['Project Name'] = df['Project Name'].str.strip()
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def norm(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv')
m = gp.Model('ProjectPortfolioOptimization')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x_vars[6] <= x_vars[1], name='prerequisite_6_requires_1')
m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')