import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
items = df['item'].tolist()
value = dict(zip(df['item'], df['value']))
weight = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value[i]}, weight={weight[i]}')
            total_weight += weight[i]
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')