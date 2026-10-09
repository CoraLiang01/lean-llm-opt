import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',')
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(float)
df['weight'] = df['weight'].astype(float)
items = df['item'].tolist()
value = dict(zip(df['item'], df['value']))
weight = dict(zip(df['item'], df['weight']))
if set(items) != set(value.keys()) or set(items) != set(weight.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')
m = gp.Model('Knapsack')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in items:
        if x[i].X > 0.5:
            print(f'Item {i}: value={value[i]}, weight={weight[i]}')
    total_weight = sum((weight[i] for i in items if x[i].X > 0.5))
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')