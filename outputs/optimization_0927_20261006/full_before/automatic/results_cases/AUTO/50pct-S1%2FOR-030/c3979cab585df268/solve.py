import gurobipy as gp
import pandas as pd
import numpy as np
import re
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
project_ids = df['Project ID'].astype(int).tolist()
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))

def norm(s):
    return re.sub('\\s+', ' ', str(s).strip().casefold())
name_to_id = {norm(row['Project Name']): int(row['Project ID']) for (_, row) in df.iterrows()}
proj_4_id = name_to_id[norm('R&D Initiative Alpha')]
proj_7_id = name_to_id[norm('Global Expansion Pilot')]
proj_6_id = name_to_id[norm('System Automation')]
proj_1_id = name_to_id[norm('Infrastructure Upgrade')]
proj_10_id = name_to_id[norm('Customer Experience Platform')]
proj_5_id = name_to_id[norm('Staff Training Program')]
required_ids = [proj_4_id, proj_7_id, proj_6_id, proj_1_id, proj_10_id, proj_5_id]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required project ID {pid} not found in project list.')
m = gp.Model('ProjectSelection')
x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
m.addConstr(x[proj_4_id] + x[proj_7_id] <= 1, name='mutual_exclusive_4_7')
m.addConstr(x[proj_6_id] <= x[proj_1_id], name='prereq_6_1')
m.addConstr(x[proj_10_id] <= x[proj_5_id], name='contingent_10_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    for i in project_ids:
        if x[i].X > 0.5:
            pname = df.loc[df['Project ID'] == i, 'Project Name'].values[0]
            print(f'Project ID {i}: {pname} | Capital: {capital[i]:.0f} k$ | NPV: {npv[i]:.0f} k$')
    total_capital = sum((capital[i] for i in project_ids if x[i].X > 0.5))
    print(f'\nTotal Capital Used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')