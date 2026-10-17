import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
required_columns = {'value', 'weight'}
missing_cols = required_columns - set(df.columns)
if missing_cols:
    raise KeyError(f'Missing required columns in value.csv: {missing_cols}')
item_id_col = None
for candidate in ['item', 'Item', 'id', 'ID']:
    if candidate in df.columns:
        item_id_col = candidate
        break
if item_id_col is not None:
    item_ids = df[item_id_col].astype(str).tolist()
else:
    item_ids = df.index.astype(str).tolist()
try:
    value_dict = {item_ids[i]: float(df.loc[i, 'value']) for i in range(len(df))}
    weight_dict = {item_ids[i]: float(df.loc[i, 'weight']) for i in range(len(df))}
except Exception as e:
    raise ValueError(f'Error converting value or weight to float: {e}')
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
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')