import pandas as pd
import gurobipy as gp
from gurobipy import GRB
project_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture4/project.csv'
df = pd.read_csv(project_path, sep=',')
if df['Project ID'].isnull().any():
    raise ValueError('Missing Project ID in project.csv')
project_ids = df['Project ID'].astype(int).tolist()
if len(set(project_ids)) != 110:
    raise ValueError('Expected 110 unique Project IDs, got %d' % len(set(project_ids)))
npv = dict(zip(df['Project ID'].astype(int), df['NPV (k$)'].astype(int)))
capital = dict(zip(df['Project ID'].astype(int), df['Capital (k$)'].astype(int)))
proj_name = dict(zip(df['Project ID'].astype(int), df['Project Name'].astype(str)))
required_ids = [1, 4, 5, 6, 7, 10]
for pid in required_ids:
    if pid not in project_ids:
        raise ValueError(f'Required Project ID {pid} not found in project.csv')

def solve_problem():
    m = gp.Model('project_selection')
    x = m.addVars(project_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
    m.setObjective(gp.quicksum((npv[i] * x[i] for i in project_ids)), GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((capital[i] * x[i] for i in project_ids)) <= 1000, name='budget')
    m.addConstr(x[4] + x[7] <= 1, name='mutual_excl_4_7')
    m.addConstr(x[6] <= x[1], name='prereq_6_1')
    m.addConstr(x[10] <= x[5], name='contingent_10_5')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for i in project_ids:
            print(f'x[{i}] {x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()