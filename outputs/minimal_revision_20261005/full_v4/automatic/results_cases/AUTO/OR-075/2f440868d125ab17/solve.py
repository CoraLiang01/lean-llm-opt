import gurobipy as gp
import pandas as pd
import numpy as np

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    if df['Project ID'].duplicated().any():
        raise ValueError('Duplicate Project IDs found in project.csv')
    projects = df['Project ID'].astype(int).tolist()
    n_projects = len(projects)
    if n_projects != 110:
        raise ValueError(f'Expected 110 projects, found {n_projects}')
    capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
    npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    required_ids = [1, 4, 5, 6, 7, 10]
    missing = [pid for pid in required_ids if pid not in projects]
    if missing:
        raise ValueError(f'Required Project IDs for constraints missing: {missing}')
    m = gp.Model('ProjectSelection')
    m.Params.MIPGap = 0.0001
    x = m.addVars(projects, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in projects)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in projects)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in projects:
            print(f'x[{i}] {x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_project_selection()