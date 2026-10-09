import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
df['item'] = df['item'].astype(str)
items = df['item'].tolist()
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch between items and parameter keys in value or weight columns.')
m = gp.Model('KnapsackDisplay')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in items:
        if x[i].X > 0.5:
            print(f'Item: {i}, Value: {value_dict[i]}, Weight: {weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in items if x[i].X > 0.5))
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')