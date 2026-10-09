import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in project.csv")
project_ids = df['Project ID'].tolist()
capital_dict = {}
npv_dict = {}
for (idx, row) in df.iterrows():
    pid = row['Project ID']
    try:
        capital = float(row['Capital (k$)'])
    except Exception as e:
        raise ValueError(f"Invalid Capital (k$) for Project ID {pid}: {row['Capital (k$)']}")
    try:
        npv = float(row['NPV (k$)'])
    except Exception as e:
        raise ValueError(f"Invalid NPV (k$) for Project ID {pid}: {row['NPV (k$)']}")
    capital_dict[pid] = capital
    npv_dict[pid] = npv
m = gp.Model('ProjectSelection')
x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[pid] * x_vars[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[pid] * x_vars[pid] for pid in project_ids)) <= 1000, name='Budget')
pid_4 = None
pid_7 = None
for (pid, name) in zip(df['Project ID'], df['Project Name']):
    if pid.strip() == '4':
        pid_4 = pid
    if pid.strip() == '7':
        pid_7 = pid
if pid_4 is None or pid_7 is None:
    raise KeyError("Could not find Project ID '4' or '7' in project.csv")
m.addConstr(x_vars[pid_4] + x_vars[pid_7] <= 1, name='MutuallyExclusive_4_7')
pid_1 = None
pid_6 = None
for (pid, name) in zip(df['Project ID'], df['Project Name']):
    if pid.strip() == '1':
        pid_1 = pid
    if pid.strip() == '6':
        pid_6 = pid
if pid_1 is None or pid_6 is None:
    raise KeyError("Could not find Project ID '1' or '6' in project.csv")
m.addConstr(x_vars[pid_6] <= x_vars[pid_1], name='Prerequisite_6_requires_1')
pid_5 = None
pid_10 = None
for (pid, name) in zip(df['Project ID'], df['Project Name']):
    if pid.strip() == '5':
        pid_5 = pid
    if pid.strip() == '10':
        pid_10 = pid
if pid_5 is None or pid_10 is None:
    raise KeyError("Could not find Project ID '5' or '10' in project.csv")
m.addConstr(x_vars[pid_10] <= x_vars[pid_5], name='Contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    for pid in project_ids:
        if x_vars[pid].X > 0.5:
            pname = df.loc[df['Project ID'] == pid, 'Project Name'].values[0]
            print(f'Project ID: {pid}, Name: {pname}, Capital: {capital_dict[pid]:.2f} k$, NPV: {npv_dict[pid]:.2f} k$')
else:
    print(f'No optimal solution found. Status: {m.status}')