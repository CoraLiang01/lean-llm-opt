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
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)} in project.csv')
capital_dict = {}
npv_dict = {}
for (idx, row) in df.iterrows():
    pid = row['Project ID']
    try:
        capital = float(row['Capital (k$)'])
        npv = float(row['NPV (k$)'])
    except Exception as e:
        raise ValueError(f'Non-numeric value in Capital or NPV for Project ID {pid}: {e}')
    capital_dict[pid] = capital
    npv_dict[pid] = npv
project_num_to_id = {}
for num in [str(i) for i in range(1, 111)]:
    if num in project_ids:
        project_num_to_id[int(num)] = num
    else:
        try:
            idx = int(num) - 1
            project_num_to_id[int(num)] = project_ids[idx]
        except Exception:
            raise ValueError(f'Cannot map Project Number {num} to Project ID in project.csv')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[pid] * y_vars[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[pid] * y_vars[pid] for pid in project_ids)) <= 1000, name='budget')
proj4_id = project_num_to_id[4]
proj7_id = project_num_to_id[7]
m.addConstr(y_vars[proj4_id] + y_vars[proj7_id] <= 1, name='mutually_exclusive_4_7')
proj6_id = project_num_to_id[6]
proj1_id = project_num_to_id[1]
m.addConstr(y_vars[proj6_id] <= y_vars[proj1_id], name='prereq_6_requires_1')
proj10_id = project_num_to_id[10]
proj5_id = project_num_to_id[5]
m.addConstr(y_vars[proj10_id] <= y_vars[proj5_id], name='contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for pid in project_ids:
        if y_vars[pid].X > 0.5:
            pname = df.loc[df['Project ID'] == pid, 'Project Name'].values[0]
            print(f'Project ID: {pid}, Name: {pname}, Capital: {capital_dict[pid]:.2f} k$, NPV: {npv_dict[pid]:.2f} k$')
            selected.append(pid)
    total_capital = sum((capital_dict[pid] for pid in selected))
    print(f'Total Capital Used: {total_capital:.2f} k$ / 1000.00 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')