import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value_param = dict(zip(df['item'], df['value']))
weight_param = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackDisplay')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    total_value = m.objVal
    total_weight = sum((weight_param[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Optimal total value: {total_value:.0f}')
    print(f'Total weight used: {total_weight}')
    print('--- Selected items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
else:
    print(f'No optimal solution found. Status: {m.status}')