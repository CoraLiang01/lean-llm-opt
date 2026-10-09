import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns:
    raise KeyError("Missing required column 'Project ID' in project.csv")
project_ids = df['Project ID'].astype(int).tolist()
if 'NPV (k$)' not in df.columns:
    raise KeyError("Missing required column 'NPV (k$)' in project.csv")
npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
if 'Capital (k$)' not in df.columns:
    raise KeyError("Missing required column 'Capital (k$)' in project.csv")
capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
name_to_id = {}
for (idx, row) in df.iterrows():
    pname = str(row['Project Name']).strip().casefold()
    pid = int(row['Project ID'])
    name_to_id[pname] = pid

def get_pid_by_name(name):
    key = name.strip().casefold()
    if key not in name_to_id:
        raise KeyError(f"Project Name '{name}' not found in project.csv")
    return name_to_id[key]
pid_1 = get_pid_by_name('Infrastructure Upgrade')
pid_4 = get_pid_by_name('R&D Initiative Alpha')
pid_5 = get_pid_by_name('Staff Training Program')
pid_6 = get_pid_by_name('System Automation')
pid_7 = get_pid_by_name('Global Expansion Pilot')
pid_10 = get_pid_by_name('Customer Experience Platform')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(y_vars[pid_4] + y_vars[pid_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(y_vars[pid_6] <= y_vars[pid_1], name='prereq_6_requires_1')
m.addConstr(y_vars[pid_10] <= y_vars[pid_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.0f} k$')
    print('Selected projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            pname = df.loc[df['Project ID'].astype(int) == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} | Capital: {capital_dict[i]} k$ | NPV: {npv_dict[i]} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total capital used: {total_capital} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')