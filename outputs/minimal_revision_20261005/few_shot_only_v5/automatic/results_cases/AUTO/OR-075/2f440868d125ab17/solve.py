import gurobipy as gp
import pandas as pd
import numpy as np

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    if 'Project ID' not in df.columns or 'Capital (k$)' not in df.columns or 'NPV (k$)' not in df.columns:
        raise ValueError('Missing required columns in project.csv')
    df['Project ID'] = df['Project ID'].astype(int)
    df = df.set_index('Project ID', drop=False)
    project_ids = list(range(1, 111))
    missing_ids = set(project_ids) - set(df.index)
    if missing_ids:
        raise ValueError(f'Missing data for Project IDs: {sorted(missing_ids)}')
    capital = df['Capital (k$)'].astype(float).to_dict()
    npv = df['NPV (k$)'].astype(float).to_dict()
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
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