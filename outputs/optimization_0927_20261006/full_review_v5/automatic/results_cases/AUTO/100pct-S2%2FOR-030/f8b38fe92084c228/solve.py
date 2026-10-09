import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns or ('Project Name' not in df.columns):
    raise KeyError('Required columns missing from project.csv')
df['Project ID'] = df['Project ID'].astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)'].astype(int)))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)'].astype(int)))

def norm(s):
    return s.strip().casefold()
name_to_id = {norm(row['Project Name']): row['Project ID'] for (_, row) in df.iterrows()}
required_names = ['r&d initiative alpha', 'global expansion pilot', 'system automation', 'infrastructure upgrade', 'customer experience platform', 'staff training program']
for pname in required_names:
    if pname not in name_to_id:
        raise ValueError(f"Project name '{pname}' not found in project.csv")
proj_4 = name_to_id['r&d initiative alpha']
proj_7 = name_to_id['global expansion pilot']
proj_6 = name_to_id['system automation']
proj_1 = name_to_id['infrastructure upgrade']
proj_10 = name_to_id['customer experience platform']
proj_5 = name_to_id['staff training program']
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')