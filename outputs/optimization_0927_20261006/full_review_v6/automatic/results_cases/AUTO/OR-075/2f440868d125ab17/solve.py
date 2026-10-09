import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
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
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
proj_1_name = 'infrastructure upgrade'
proj_4_name = 'r&d initiative alpha'
proj_5_name = 'staff training program'
proj_6_name = 'system automation'
proj_7_name = 'global expansion pilot'
proj_10_name = 'customer experience platform'
try:
    proj_1_id = name_to_id[norm(proj_1_name)]
    proj_4_id = name_to_id[norm(proj_4_name)]
    proj_5_id = name_to_id[norm(proj_5_name)]
    proj_6_id = name_to_id[norm(proj_6_name)]
    proj_7_id = name_to_id[norm(proj_7_name)]
    proj_10_id = name_to_id[norm(proj_10_name)]
except KeyError as e:
    raise KeyError(f'Could not find required project name in CSV: {e}')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_excl_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
            selected.append(i)
    print(f'\nTotal selected: {len(selected)} projects')
    total_cap = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_cap} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')