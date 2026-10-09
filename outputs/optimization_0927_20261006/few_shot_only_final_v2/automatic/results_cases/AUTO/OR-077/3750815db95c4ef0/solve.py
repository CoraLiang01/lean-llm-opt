import gurobipy as gp
import pandas as pd
import numpy as np
import re
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df_value = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
items = df_value['item'].tolist()
try:
    value_dict = {}
    weight_dict = {}
    for (idx, row) in df_value.iterrows():
        item_id = str(row['item'])
        try:
            value = float(row['value'])
            weight = float(row['weight'])
        except Exception as e:
            raise ValueError(f"Non-numeric value or weight for item '{item_id}': {e}")
        value_dict[item_id] = value
        weight_dict[item_id] = weight
except KeyError as e:
    raise KeyError(f'Missing required column in value.csv: {e}')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[item] * x_vars[item] for item in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[item] * x_vars[item] for item in items)) <= 15, name='WeightLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0.0
    for item in items:
        if x_vars[item].X > 0.5:
            print(f'Item: {item}, Value: {value_dict[item]}, Weight: {weight_dict[item]}')
            total_weight += weight_dict[item]
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')