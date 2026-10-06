import gurobipy as gp
import pandas as pd
import numpy as np

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    if 'Project ID' not in df.columns or 'NPV (k$)' not in df.columns or 'Capital (k$)' not in df.columns:
        raise KeyError('Required columns missing from project.csv')
    project_ids = df['Project ID'].astype(int).tolist()
    n_projects = len(project_ids)
    if n_projects != 110:
        raise ValueError(f'Expected 110 projects, found {n_projects}')
    npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
    for i in project_ids:
        if i not in npv or i not in capital:
            raise KeyError(f'Missing NPV or Capital for Project ID {i}')
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    if 4 in project_ids and 7 in project_ids:
        m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
    else:
        raise KeyError('Project 4 or 7 not found in project.csv')
    if 6 in project_ids and 1 in project_ids:
        m.addConstr(x[6] <= x[1], name='prereq_6_1')
    else:
        raise KeyError('Project 6 or 1 not found in project.csv')
    if 10 in project_ids and 5 in project_ids:
        m.addConstr(x[10] <= x[5], name='contingent_10_5')
    else:
        raise KeyError('Project 10 or 5 not found in project.csv')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in project_ids:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_project_selection()