import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
item_ids = df['item'].astype(int).tolist()
try:
    value_param = df.set_index(df['item'].astype(int))['value'].astype(int).to_dict()
    weight_param = df.set_index(df['item'].astype(int))['weight'].astype(int).to_dict()
except Exception as e:
    raise ValueError(f"Error converting 'value' or 'weight' columns to int: {e}")
if set(item_ids) != set(value_param.keys()) or set(item_ids) != set(weight_param.keys()):
    raise ValueError('Mismatch in item indices and value/weight parameter keys.')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
    total_weight = sum((weight_param[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight}')
else:
    print(f'No optimal solution found. Status: {m.status}')