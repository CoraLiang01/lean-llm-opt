import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column: 'Project ID'")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError("Missing required columns: 'NPV (k$)' and/or 'Capital (k$)'")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
if 'Project Name' not in df.columns:
    raise KeyError("Missing required column: 'Project Name'")

def norm_str(s):
    return s.strip().casefold()
name_to_id = {}
for (idx, row) in df.iterrows():
    name_to_id[norm_str(row['Project Name'])] = int(row['Project ID'])
id_rnd_alpha = name_to_id.get(norm_str('R&D Initiative Alpha'))
id_global_expansion = name_to_id.get(norm_str('Global Expansion Pilot'))
id_system_automation = name_to_id.get(norm_str('System Automation'))
id_infrastructure_upgrade = name_to_id.get(norm_str('Infrastructure Upgrade'))
id_customer_experience = name_to_id.get(norm_str('Customer Experience Platform'))
id_staff_training = name_to_id.get(norm_str('Staff Training Program'))
required_ids = {'R&D Initiative Alpha': id_rnd_alpha, 'Global Expansion Pilot': id_global_expansion, 'System Automation': id_system_automation, 'Infrastructure Upgrade': id_infrastructure_upgrade, 'Customer Experience Platform': id_customer_experience, 'Staff Training Program': id_staff_training}
for (name, pid) in required_ids.items():
    if pid is None:
        raise ValueError(f"Could not find Project ID for required project: '{name}'")
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x_vars[id_rnd_alpha] + x_vars[id_global_expansion] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x_vars[id_system_automation] <= x_vars[id_infrastructure_upgrade], name='prereq_6_1')
m.addConstr(x_vars[id_customer_experience] <= x_vars[id_staff_training], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if x_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'].astype(int) == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital_dict[i]} k$, NPV: {npv_dict[i]} k$)')
    total_capital = sum((capital_dict[i] for i in project_ids if x_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')