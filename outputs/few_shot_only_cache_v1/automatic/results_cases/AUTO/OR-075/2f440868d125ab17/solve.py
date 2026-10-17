import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
    raise KeyError('Required columns missing from project.csv')
project_ids = df['Project ID'].tolist()
capital_dict = {}
npv_dict = {}
for idx, row in df.iterrows():
    pid = str(row['Project ID']).strip()
    try:
        capital = float(row['Capital (k$)'])
        npv = float(row['NPV (k$)'])
    except Exception as e:
        raise ValueError(f'Non-numeric value in Capital or NPV for Project ID {pid}: {e}')
    capital_dict[pid] = capital
    npv_dict[pid] = npv
required_ids = ['1', '4', '5', '6', '7', '10']
for rid in required_ids:
    if rid not in project_ids:
        raise KeyError(f'Project ID {rid} required by constraints is missing from project.csv')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[pid] * y_vars[pid] for pid in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[pid] * y_vars[pid] for pid in project_ids)) <= 1000, name='Budget')
m.addConstr(y_vars['4'] + y_vars['7'] <= 1, name='MutuallyExclusive_4_7')
m.addConstr(y_vars['6'] <= y_vars['1'], name='Prerequisite_6_requires_1')
m.addConstr(y_vars['10'] <= y_vars['5'], name='Contingent_10_requires_5')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total expected NPV: {m.objVal:.2f} k$')
    print('\n--- Selected Projects ---')
    selected = []
    for pid in project_ids:
        if y_vars[pid].X > 0.5:
            pname = df.loc[df['Project ID'] == pid, 'Project Name'].values[0]
            print(f'  Project {pid}: {pname} | Capital: {capital_dict[pid]:.0f} k$ | NPV: {npv_dict[pid]:.0f} k$')
            selected.append(pid)
    print(f'\nTotal selected: {len(selected)} projects')
    total_capital = sum((capital_dict[pid] for pid in selected))
    print(f'Total capital used: {total_capital:.2f} k$ / 1000 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')