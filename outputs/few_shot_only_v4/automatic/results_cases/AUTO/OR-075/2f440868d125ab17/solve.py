import gurobipy as gp
import pandas as pd
import numpy as np
import re
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
project_ids = list(range(1, 111))
if not {'Project ID', 'NPV (k$)', 'Capital (k$)'}.issubset(df.columns):
    raise KeyError("project.csv must contain columns: 'Project ID', 'NPV (k$)', 'Capital (k$)'")

def parse_int(x):
    try:
        return int(str(x).strip())
    except Exception:
        raise ValueError(f'Invalid Project ID: {x}')
df['Project ID_int'] = df['Project ID'].apply(parse_int)
npv_dict = {}
capital_dict = {}
for _, row in df.iterrows():
    pid = int(row['Project ID_int'])
    if pid in project_ids:
        try:
            npv = float(str(row['NPV (k$)']).replace(',', '').strip())
            capital = float(str(row['Capital (k$)']).replace(',', '').strip())
        except Exception:
            raise ValueError(f'Invalid NPV or Capital for Project ID {pid}')
        npv_dict[pid] = npv
        capital_dict[pid] = capital
missing_npv = [i for i in project_ids if i not in npv_dict]
missing_capital = [i for i in project_ids if i not in capital_dict]
if missing_npv or missing_capital:
    raise ValueError(f'Missing NPV or Capital data for Project IDs: {set(missing_npv + missing_capital)}')
m = gp.Model('ProjectSelection')
y_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((npv_dict[i] * y_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((capital_dict[i] * y_vars[i] for i in project_ids)) <= 1000, name='budget')
if 4 in project_ids and 7 in project_ids:
    m.addConstr(y_vars[4] + y_vars[7] <= 1, name='mutual_excl_4_7')
else:
    raise ValueError('Project 4 and/or 7 not found in project_ids.')
if 6 in project_ids and 1 in project_ids:
    m.addConstr(y_vars[6] <= y_vars[1], name='prereq_6_1')
else:
    raise ValueError('Project 6 and/or 1 not found in project_ids.')
if 10 in project_ids and 5 in project_ids:
    m.addConstr(y_vars[10] <= y_vars[5], name='contingent_10_5')
else:
    raise ValueError('Project 10 and/or 5 not found in project_ids.')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total NPV: {m.objVal:.2f} k$')
    print('Selected Projects:')
    for i in project_ids:
        if y_vars[i].X > 0.5:
            print(f'  Project {i}: NPV={npv_dict[i]:.2f} k$, Capital={capital_dict[i]:.2f} k$')
    total_capital = sum((capital_dict[i] for i in project_ids if y_vars[i].X > 0.5))
    print(f'Total Capital Used: {total_capital:.2f} k$ / 1000.00 k$')
else:
    print(f'No optimal solution found. Status: {m.status}')