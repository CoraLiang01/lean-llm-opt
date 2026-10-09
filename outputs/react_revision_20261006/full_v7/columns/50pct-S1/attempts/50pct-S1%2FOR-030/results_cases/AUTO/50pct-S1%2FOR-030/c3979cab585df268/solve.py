import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',', dtype=str, keep_default_na=False)
for col in ['Project ID', 'Capital (k$)', 'NPV (k$)']:
    if not set(df[col]).issubset(set(map(str, range(1000000)))):
        raise ValueError(f'Non-numeric value found in column {col}')
    df[col] = df[col].astype(int)
project_ids = df['Project ID'].tolist()
project_ids_set = set(project_ids)
if len(project_ids) != 110:
    raise ValueError(f'Expected 110 projects, found {len(project_ids)}')
npv = dict(zip(df['Project ID'], df['NPV (k$)']))
capital = dict(zip(df['Project ID'], df['Capital (k$)']))
if set(npv.keys()) != project_ids_set or set(capital.keys()) != project_ids_set:
    raise ValueError('Mismatch in project IDs for NPV or Capital')
name_to_id = {}
for (idx, row) in df.iterrows():
    name_to_id[row['Project Name'].strip().casefold()] = row['Project ID']
required_names = ['r&d initiative alpha', 'global expansion pilot', 'system automation', 'infrastructure upgrade', 'customer experience platform', 'staff training program']
for name in required_names:
    if name not in name_to_id:
        raise ValueError(f"Project name '{name}' not found in data")
proj_4 = name_to_id['r&d initiative alpha']
proj_7 = name_to_id['global expansion pilot']
proj_6 = name_to_id['system automation']
proj_1 = name_to_id['infrastructure upgrade']
proj_10 = name_to_id['customer experience platform']
proj_5 = name_to_id['staff training program']

def solve_project_selection(project_ids, npv, capital, proj_4, proj_7, proj_6, proj_1, proj_10, proj_5):
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[proj_4] + x_vars[proj_7] <= 1, name='mutual_4_7')
    m.addConstr(x_vars[proj_6] <= x_vars[proj_1], name='prereq_6_1')
    m.addConstr(x_vars[proj_10] <= x_vars[proj_5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_project_selection(project_ids=project_ids, npv=npv, capital=capital, proj_4=proj_4, proj_7=proj_7, proj_6=proj_6, proj_1=proj_1, proj_10=proj_10, proj_5=proj_5)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')