import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)', 'Project Name'}.issubset(df.columns):
    raise KeyError('Missing required columns in project.csv')
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))
name_to_id = {name.strip().casefold(): pid for name, pid in zip(df['Project Name'], df['Project ID'])}

def get_project_id_by_name(target_name):
    key = target_name.strip().casefold()
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in project.csv")
    return name_to_id[key]
proj4_id = get_project_id_by_name('R&D Initiative Alpha')
proj7_id = get_project_id_by_name('Global Expansion Pilot')
proj6_id = get_project_id_by_name('System Automation')
proj1_id = get_project_id_by_name('Infrastructure Upgrade')
proj10_id = get_project_id_by_name('Customer Experience Platform')
proj5_id = get_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[proj4_id] + x_vars[proj7_id] <= 1, name='mutually_exclusive_4_7')
m.addConstr(x_vars[proj6_id] <= x_vars[proj1_id], name='prerequisite_6_requires_1')
m.addConstr(x_vars[proj10_id] <= x_vars[proj5_id], name='contingent_10_requires_5')
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