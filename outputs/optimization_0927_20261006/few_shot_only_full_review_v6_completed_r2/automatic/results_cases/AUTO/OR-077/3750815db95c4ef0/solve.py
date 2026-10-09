import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['item', 'value', 'weight']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in value.csv")
items = df['item'].tolist()
try:
    value_dict = dict(zip(df['item'], pd.to_numeric(df['value'], errors='raise')))
    weight_dict = dict(zip(df['item'], pd.to_numeric(df['weight'], errors='raise')))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to numeric: {e}')
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in items and value/weight keys after conversion.')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0.0
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item: {i}, Value: {value_dict[i]}, Weight: {weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')