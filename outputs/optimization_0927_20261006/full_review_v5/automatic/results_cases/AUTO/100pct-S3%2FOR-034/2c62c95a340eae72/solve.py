import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Missing required columns in value.csv: 'item', 'value', or 'weight'.")
item_ids = df['item'].astype(str).tolist()
try:
    value_dict = dict(zip(item_ids, df['value'].astype(int)))
    weight_dict = dict(zip(item_ids, df['weight'].astype(int)))
except Exception as e:
    raise ValueError(f"Error converting 'value' or 'weight' columns to int: {e}")
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='WeightLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')