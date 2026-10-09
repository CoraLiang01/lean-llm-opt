import gurobipy as gp
import pandas as pd
import numpy as np
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',', dtype=str, keep_default_na=False)
if not {'Project ID', 'Capital (k$)', 'NPV (k$)', 'Project Name'}.issubset(df.columns):
    raise ValueError('Missing required columns in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
project_names = df.set_index(df['Project ID'].astype(int))['Project Name'].to_dict()
capital = df.set_index(df['Project ID'].astype(int))['Capital (k$)'].astype(int).to_dict()
npv = df.set_index(df['Project ID'].astype(int))['NPV (k$)'].astype(int).to_dict()
for i in project_ids:
    if i not in capital or i not in npv:
        raise ValueError(f'Missing capital or NPV for project {i}')
special_projects = {'prereq_1': 1, 'mutual_4': 4, 'mutual_7': 7, 'prereq_6': 6, 'contingent_5': 5, 'contingent_10': 10}
for (key, pid) in special_projects.items():
    if pid not in project_ids:
        raise ValueError(f"Required project ID {pid} for constraint '{key}' not found in data.")

def solve_problem():
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[special_projects['mutual_4']] + x_vars[special_projects['mutual_7']] <= 1, name='mutual_exclusive_4_7')
    m.addConstr(x_vars[special_projects['prereq_6']] <= x_vars[special_projects['prereq_1']], name='prereq_6_requires_1')
    m.addConstr(x_vars[special_projects['contingent_10']] <= x_vars[special_projects['contingent_5']], name='contingent_10_requires_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')