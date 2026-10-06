import gurobipy as gp
import pandas as pd
import numpy as np

def solve_knapsack():
    path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others3/value.csv'
    df = pd.read_csv(path, sep=',')
    if df['item'].isnull().any():
        raise ValueError('Missing item identifiers in value.csv')
    items = df['item'].astype(int).tolist()
    value_dict = dict(zip(df['item'].astype(int), df['value'].astype(float)))
    weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(float)))
    if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
        raise ValueError('Mismatch in item keys between index and value/weight columns')
    capacity = 15.0
    m = gp.Model('Knapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= capacity, name='weight_limit')
    m.optimize()
    if m.status == gp.GRB.OPTIMAL:
        print(f'ObjVal: {m.objVal}')
        for i in items:
            print(f'{x[i].VarName} {x[i].X}')
    else:
        print(f'Solver status: {m.status}')
    return m
m = solve_knapsack()