import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/50pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',')
items = df['item'].astype(int).tolist()
value = df.set_index('item')['value'].astype(float).to_dict()
weight = df.set_index('item')['weight'].astype(float).to_dict()
missing_value = set(items) - set(value.keys())
missing_weight = set(items) - set(weight.keys())
if missing_value or missing_weight:
    raise ValueError(f'Missing value or weight for items: {missing_value | missing_weight}')
m = gp.Model('KnapsackDisplaySelection')
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