import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['item'] = df['item'].astype(int)
items = df['item'].tolist()
try:
    value_param = df.set_index('item')['value'].astype(int).to_dict()
    weight_param = df.set_index('item')['weight'].astype(int).to_dict()
except KeyError as e:
    raise KeyError(f'Missing required column in value.csv: {e}')
if set(items) != set(value_param.keys()) or set(items) != set(weight_param.keys()):
    raise ValueError('Mismatch in item indices between value, weight, and item columns.')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
            total_weight += weight_param[i]
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')