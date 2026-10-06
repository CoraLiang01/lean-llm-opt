import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
    raise KeyError('Missing required columns in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got %d' % len(set(project_ids)))
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
for pid in project_ids:
    if pid not in npv or pid not in capital:
        raise ValueError(f'Missing NPV or Capital for Project ID {pid}')
special_ids = [1, 4, 5, 6, 7, 10]
for sid in special_ids:
    if sid not in project_ids:
        raise ValueError(f'Required Project ID {sid} not found in project.csv')

def solve_problem():
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')