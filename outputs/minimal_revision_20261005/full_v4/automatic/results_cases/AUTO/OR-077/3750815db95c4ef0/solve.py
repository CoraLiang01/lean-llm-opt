import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
if not set(['item', 'value', 'weight']).issubset(df.columns):
    raise ValueError('Missing required columns in value.csv')
items = df['item'].astype(int).tolist()
value_dict = dict(zip(df['item'].astype(int), df['value'].astype(int)))
weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(int)))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item keys between value and weight columns.')
capacity = 15

def solve_knapsack(items, value_dict, weight_dict, capacity):
    m = gp.Model('Knapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= capacity, name='weight_limit')
    m.optimize()
    return m
m = solve_knapsack(items, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')