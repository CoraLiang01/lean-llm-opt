import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/100pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(csv_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
name_to_id = {re.sub('\\s+', ' ', n.casefold().strip()): pid for n, pid in zip(df['Project Name'], df['Project ID'])}

def get_pid_by_name(target_name):
    key = re.sub('\\s+', ' ', target_name.casefold().strip())
    if key not in name_to_id:
        raise ValueError(f"Project name '{target_name}' not found in CSV.")
    return int(name_to_id[key])
pid_1 = get_pid_by_name('Infrastructure Upgrade')
pid_4 = get_pid_by_name('R&D Initiative Alpha')
pid_5 = get_pid_by_name('Staff Training Program')
pid_6 = get_pid_by_name('System Automation')
pid_7 = get_pid_by_name('Global Expansion Pilot')
pid_10 = get_pid_by_name('Customer Experience Platform')
for pid in [pid_1, pid_4, pid_5, pid_6, pid_7, pid_10]:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project_ids.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[pid_4] + x[pid_7] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[pid_6] <= x[pid_1], name='prereq_6_requires_1')
m.addConstr(x[pid_10] <= x[pid_5], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected Projects:')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'  Project ID {i}: {pname} (Capital: {capital[i]:.0f} k$, NPV: {npv[i]:.0f} k$)')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'Total Capital Used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')