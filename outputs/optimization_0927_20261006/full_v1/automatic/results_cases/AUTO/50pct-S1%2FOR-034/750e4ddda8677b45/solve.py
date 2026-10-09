import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
item_ids = df['item'].astype(int).tolist()
try:
    value_param = {int(row['item']): int(row['value']) for (_, row) in df.iterrows()}
    weight_param = {int(row['item']): int(row['weight']) for (_, row) in df.iterrows()}
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to int: {e}')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
            total_weight += weight_param[i]
    print(f'Total weight: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')