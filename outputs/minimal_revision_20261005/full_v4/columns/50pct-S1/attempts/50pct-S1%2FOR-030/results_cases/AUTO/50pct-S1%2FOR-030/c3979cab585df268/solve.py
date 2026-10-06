import gurobipy as gp
import pandas as pd
import numpy as np
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
project_ids = df['Project ID'].astype(int).tolist()
n_projects = len(project_ids)
if n_projects != 110:
    raise ValueError(f'Expected 110 projects, found {n_projects} in CSV.')
npv = df.set_index('Project ID')['NPV (k$)'].astype(float).to_dict()
capital = df.set_index('Project ID')['Capital (k$)'].astype(float).to_dict()
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required project ID {pid} not found in project list.')

def solve_problem():
    m = gp.Model('ProjectSelection')
    x = m.addVars(project_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')