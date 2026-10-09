import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
df['item'] = df['item'].astype(int)
items = df['item'].tolist()
try:
    value_dict = dict(zip(df['item'], df['value'].astype(int)))
    weight_dict = dict(zip(df['item'], df['weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to int: {e}')
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item keys between index set and value/weight dictionaries.')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')