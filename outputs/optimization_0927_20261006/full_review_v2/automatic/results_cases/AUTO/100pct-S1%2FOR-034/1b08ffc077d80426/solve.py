import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {norm_col(col): col for col in df.columns}
required_cols = ['item', 'value', 'weight']
for rc in required_cols:
    if rc not in col_map:
        raise KeyError(f"Required column '{rc}' not found in CSV.")
item_col = col_map['item']
value_col = col_map['value']
weight_col = col_map['weight']
df['item_id'] = df[item_col].astype(str).str.strip()
items = df['item_id'].tolist()
try:
    value_param = dict(zip(df['item_id'], df[value_col].apply(lambda x: int(x.strip()))))
    weight_param = dict(zip(df['item_id'], df[weight_col].apply(lambda x: int(x.strip()))))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to int: {e}')
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
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
    total_weight = sum((weight_param[i] for i in items if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')