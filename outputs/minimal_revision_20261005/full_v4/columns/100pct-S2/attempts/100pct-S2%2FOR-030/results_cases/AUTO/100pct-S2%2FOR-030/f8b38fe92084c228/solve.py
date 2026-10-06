import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Mixture_testing/Mixture4/project.csv', sep=',')
    project_ids = df['Project ID'].astype(int).tolist()
    n_projects = len(project_ids)
    if n_projects != 110:
        raise ValueError(f'Expected 110 projects, found {n_projects}')
    npv_dict = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(float)))
    capital_dict = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(float)))
    for pid in project_ids:
        if pid not in npv_dict or pid not in capital_dict:
            raise ValueError(f'Missing NPV or Capital for Project ID {pid}')
    required_ids = [1, 4, 5, 6, 7, 10]
    for rid in required_ids:
        if rid not in project_ids:
            raise ValueError(f'Required Project ID {rid} not found in data')
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv_dict[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital_dict[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in project_ids:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_problem()