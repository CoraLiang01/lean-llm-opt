import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
df['Project ID'] = df['Project ID'].astype(int)
df['Capital (k$)'] = df['Capital (k$)'].astype(int)
df['NPV (k$)'] = df['NPV (k$)'].astype(int)
df['Project Name_norm'] = df['Project Name'].str.strip().str.casefold()
project_ids = df['Project ID'].tolist()
capital_dict = dict(zip(df['Project ID'], df['Capital (k$)']))
npv_dict = dict(zip(df['Project ID'], df['NPV (k$)']))

def find_project_id_by_name(target_name):
    norm_name = target_name.strip().casefold()
    matches = df[df['Project Name_norm'] == norm_name]
    if len(matches) != 1:
        raise ValueError(f"Could not uniquely identify project with name '{target_name}'. Found {len(matches)} matches.")
    return int(matches.iloc[0]['Project ID'])
id_rnd_alpha = find_project_id_by_name('R&D Initiative Alpha')
id_global_expansion = find_project_id_by_name('Global Expansion Pilot')
id_system_automation = find_project_id_by_name('System Automation')
id_infra_upgrade = find_project_id_by_name('Infrastructure Upgrade')
id_cust_exp_platform = find_project_id_by_name('Customer Experience Platform')
id_staff_training = find_project_id_by_name('Staff Training Program')
m = gp.Model('ProjectSelection_NPV_Max')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[id_rnd_alpha] + x_vars[id_global_expansion] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[id_system_automation] <= x_vars[id_infra_upgrade], name='prereq_6_requires_1')
m.addConstr(x_vars[id_cust_exp_platform] <= x_vars[id_staff_training], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    selected = []
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital_dict[i]} k$, NPV: {npv_dict[i]} k$)')
            selected.append(i)
    total_capital = sum((capital_dict[i] for i in selected))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')