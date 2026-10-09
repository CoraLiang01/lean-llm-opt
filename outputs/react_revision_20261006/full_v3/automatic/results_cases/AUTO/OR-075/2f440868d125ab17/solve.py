import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
required_cols = ['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f'Missing required column: {col}')
project_ids = df['Project ID'].astype(int).tolist()
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
if set(project_ids) != set(npv.keys()) or set(project_ids) != set(capital.keys()):
    raise ValueError('Mismatch in project IDs between data and parameter dictionaries.')
special_ids = [1, 4, 5, 6, 7, 10]
for pid in special_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in data.')

def solve_problem():
    m = gp.Model('ProjectSelection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')