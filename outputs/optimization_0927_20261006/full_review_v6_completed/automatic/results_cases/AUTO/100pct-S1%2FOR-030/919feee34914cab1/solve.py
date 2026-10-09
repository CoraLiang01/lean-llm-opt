import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(float)
df['NPV (k$)'] = df['NPV (k$)'].astype(float)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(target_name):
    norm_name = target_name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm_name]['Project ID'].tolist()
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found: {matches}")
    return matches[0]
proj_1_id = find_project_id_by_name('Infrastructure Upgrade')
proj_4_id = find_project_id_by_name('R&D Initiative Alpha')
proj_5_id = find_project_id_by_name('Staff Training Program')
proj_6_id = find_project_id_by_name('System Automation')
proj_7_id = find_project_id_by_name('Global Expansion Pilot')
proj_10_id = find_project_id_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj_4_id] + x_vars[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[proj_6_id] <= x_vars[proj_1_id], name='prereq_6_requires_1')
m.addConstr(x_vars[proj_10_id] <= x_vars[proj_5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital_dict[i]:.0f} k$, NPV: {npv_dict[i]:.0f} k$)')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')