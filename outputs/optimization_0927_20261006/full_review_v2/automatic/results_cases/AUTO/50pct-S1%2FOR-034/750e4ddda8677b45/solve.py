import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {norm_col(col): col for col in df.columns}
item_col = col_map['item']
value_col = col_map['value']
weight_col = col_map['weight']
df_items = df.copy()
df_items['item_id'] = df_items[item_col].astype(int)
item_ids = df_items['item_id'].tolist()
try:
    value_dict = dict(zip(df_items['item_id'], df_items[value_col].astype(int)))
    weight_dict = dict(zip(df_items['item_id'], df_items[weight_col].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to int: {e}')
if set(value_dict.keys()) != set(item_ids) or set(weight_dict.keys()) != set(item_ids):
    raise ValueError('Mismatch in item index coverage for value or weight.')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')