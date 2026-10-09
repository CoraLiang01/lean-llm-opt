import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
if not {'item', 'value', 'weight'}.issubset(df.columns):
    raise KeyError("CSV file must contain columns: 'item', 'value', 'weight'.")
items = df['item'].astype(int).tolist()
if len(set(items)) != len(items):
    raise ValueError("Duplicate item identifiers found in 'item' column.")
value_dict = dict(zip(df['item'].astype(int), df['value'].astype(int)))
weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(int)))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item identifiers between index set and parameter dictionaries.')
m = gp.Model('Knapsack')
x = m.addVars(items, vtype=gp.GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in items:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')