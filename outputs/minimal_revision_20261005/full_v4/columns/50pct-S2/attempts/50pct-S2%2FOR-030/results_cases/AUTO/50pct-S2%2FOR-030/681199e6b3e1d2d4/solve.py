import gurobipy as gp
import pandas as pd
import numpy as np
import re

def solve_project_selection():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    project_ids = list(range(1, 111))
    df_ids = set(df['Project ID'].astype(int))
    missing_ids = set(project_ids) - df_ids
    if missing_ids:
        raise ValueError(f'Missing required Project IDs in CSV: {sorted(missing_ids)}')
    npv = df.set_index('Project ID')['NPV (k$)'].astype(int).to_dict()
    capital = df.set_index('Project ID')['Capital (k$)'].astype(int).to_dict()
    for i in project_ids:
        if i not in npv or i not in capital:
            raise ValueError(f'Missing NPV or Capital for Project ID {i}')
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((x[i] * npv[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((x[i] * capital[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in project_ids:
            print(f'x[{i}] {x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_project_selection()