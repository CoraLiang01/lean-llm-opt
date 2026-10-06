import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',')
if df['item'].isnull().any():
    raise ValueError('Missing item identifiers in value.csv')
items = df['item'].astype(int).tolist()
if df['value'].isnull().any():
    raise ValueError('Missing value coefficients in value.csv')
if df['weight'].isnull().any():
    raise ValueError('Missing weight coefficients in value.csv')
value = dict(zip(df['item'].astype(int), df['value'].astype(float)))
weight = dict(zip(df['item'].astype(int), df['weight'].astype(float)))
if set(items) != set(value.keys()) or set(items) != set(weight.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')

def solve_knapsack(items, value, weight, weight_limit):
    m = gp.Model('Knapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= weight_limit, name='weight_limit')
    m.optimize()
    return m
m = solve_knapsack(items, value, weight, weight_limit=15)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')