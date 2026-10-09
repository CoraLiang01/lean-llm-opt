import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',', dtype=str, keep_default_na=False)
    if not set(['Project ID', 'Project Name', 'Capital (k$)', 'NPV (k$)']).issubset(df.columns):
        raise ValueError('Missing required columns in project.csv')
    df['Project ID'] = df['Project ID'].astype(str).str.strip()
    try:
        project_ids = df['Project ID'].astype(int).tolist()
    except Exception as e:
        raise ValueError('Project ID column must be convertible to int') from e
    project_names = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
    try:
        capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
        npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    except Exception as e:
        raise ValueError('Capital (k$) and NPV (k$) columns must be convertible to float') from e
    required_projects = [1, 4, 5, 6, 7, 10]
    for pid in required_projects:
        if pid not in project_ids:
            raise ValueError(f'Required project ID {pid} not found in project.csv')
    x_vars = {}
    m = gp.Model('ProjectSelection')
    x_vars = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x_vars[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x_vars[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x_vars[4] + x_vars[7] <= 1, name='mutual_4_7')
    m.addConstr(x_vars[6] <= x_vars[1], name='prereq_6_1')
    m.addConstr(x_vars[10] <= x_vars[5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')