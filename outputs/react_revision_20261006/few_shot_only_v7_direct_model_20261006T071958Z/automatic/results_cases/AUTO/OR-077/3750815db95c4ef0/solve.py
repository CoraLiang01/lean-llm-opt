import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['item', 'value', 'weight']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Missing required column '{col}' in value.csv")
items = df['item'].tolist()
if len(set(items)) != len(items):
    raise ValueError('Duplicate item identifiers found in value.csv')
try:
    value_param = dict(zip(items, df['value'].astype(float)))
    weight_param = dict(zip(items, df['weight'].astype(float)))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to float: {e}')
if set(value_param.keys()) != set(items) or set(weight_param.keys()) != set(items):
    raise ValueError('Mismatch in item keys for value or weight parameters.')
capacity = 15.0

def solve_knapsack(items, value_param, weight_param, capacity):
    m = gp.Model('Knapsack')
    x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in items)) <= capacity, name='weight_limit')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_knapsack(items, value_param, weight_param, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')