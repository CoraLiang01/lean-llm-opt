import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
if not {'item', 'value', 'weight'}.issubset(df.columns):
    raise KeyError("CSV must contain columns: 'item', 'value', 'weight'")
items = df['item'].astype(str).tolist()
try:
    value_param = dict(zip(df['item'].astype(str), df['value'].astype(float)))
    weight_param = dict(zip(df['item'].astype(str), df['weight'].astype(float)))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to float: {e}')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item: {i}, Value: {value_param[i]}, Weight: {weight_param[i]}')
    total_weight = sum((weight_param[i] for i in items if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')