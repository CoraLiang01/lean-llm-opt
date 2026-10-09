import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'item', 'value', or 'weight' not found in value.csv.")
item_ids = df['item'].astype(str).tolist()
try:
    value_dict = dict(zip(item_ids, df['value'].astype(float)))
    weight_dict = dict(zip(item_ids, df['weight'].astype(float)))
except Exception as e:
    raise ValueError(f"Error converting 'value' or 'weight' columns to float: {e}")
if set(value_dict.keys()) != set(item_ids) or set(weight_dict.keys()) != set(item_ids):
    raise ValueError('Mismatch in item IDs between value and weight columns.')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='TotalWeightLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0.0
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')