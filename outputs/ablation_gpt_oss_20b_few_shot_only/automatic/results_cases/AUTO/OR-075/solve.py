import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['Project ID'] = df['Project ID'].str.strip()
    df['Capital (k$)'] = df['Capital (k$)'].str.strip()
    df['NPV (k$)'] = df['NPV (k$)'].str.strip()
    df['Project ID'] = df['Project ID'].astype(int)
    df['Capital (k$)'] = df['Capital (k$)'].astype(float)
    df['NPV (k$)'] = df['NPV (k$)'].astype(float)
    c = dict(zip(df['Project ID'], df['Capital (k$)']))
    n = dict(zip(df['Project ID'], df['NPV (k$)']))
    P = list(range(1, 111))
    for p in P:
        if p not in c or p not in n:
            raise ValueError(f'Missing data for project {p}')
    m = gp.Model('ProjectSelection')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(P, vtype=GRB.BINARY, name='')
    m.setObjective(gp.quicksum((n[p] * x[p] for p in P)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((c[p] * x[p] for p in P)) <= 1000, name='Budget')
    m.addConstr(x[4] + x[7] <= 1, name='MutualExcl')
    m.addConstr(x[6] <= x[1], name='Prereq')
    m.addConstr(x[10] <= x[5], name='Contingent')
    m.optimize()
    return m
m = solve_problem()