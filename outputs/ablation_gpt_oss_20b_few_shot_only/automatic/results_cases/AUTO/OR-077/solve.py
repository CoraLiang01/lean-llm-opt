import gurobipy
from gurobipy import GRB
import pandas as pd

def solve_problem():
    path = 'Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df['item'] = df['item'].astype(int)
    df['value'] = df['value'].astype(float)
    df['weight'] = df['weight'].astype(float)
    ids = df['item'].tolist()
    v = dict(zip(df['item'], df['value']))
    w = dict(zip(df['item'], df['weight']))
    if set(ids) != set(v.keys()) or set(ids) != set(w.keys()):
        missing = set(ids) - set(v.keys())
        raise ValueError(f'Missing value data for items: {missing}')
    m = gurobipy.Model('Knapsack')
    m.setParam('MIPGap', 1e-06)
    x = m.addVars(ids, vtype=GRB.BINARY, name='')
    m.setObjective(gurobipy.quicksum((v[i] * x[i] for i in ids)), GRB.MAXIMIZE)
    m.addConstr(gurobipy.quicksum((w[i] * x[i] for i in ids)) <= 15, name='Weight')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'Optimal objective: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName} {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()