import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
items = df['item'].astype(int).tolist()
if len(set(items)) != len(items):
    raise ValueError('Duplicate item identifiers found in value.csv.')
value_dict = dict(zip(df['item'].astype(int), df['value'].astype(int)))
weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(int)))
for i in items:
    if i not in value_dict or i not in weight_dict:
        raise ValueError(f'Missing value or weight for item {i}.')
m = gp.Model('Knapsack')
m.Params.MIPGap = 0.0001
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in items:
        print(f'x[{i}] {x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')