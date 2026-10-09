import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Missing required columns in value.csv: 'item', 'value', or 'weight'.")
item_ids = df['item'].tolist()
try:
    value_param = {item_id: int(df.loc[df['item'] == item_id, 'value'].iloc[0]) for item_id in item_ids}
    weight_param = {item_id: int(df.loc[df['item'] == item_id, 'weight'].iloc[0]) for item_id in item_ids}
except Exception as e:
    raise ValueError(f'Error converting value or weight to int: {e}')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[item_id] * x_vars[item_id] for item_id in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[item_id] * x_vars[item_id] for item_id in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for item_id in item_ids:
        if x_vars[item_id].X > 0.5:
            print(f'Item {item_id}: value={value_param[item_id]}, weight={weight_param[item_id]}')
    total_weight = sum((weight_param[item_id] for item_id in item_ids if x_vars[item_id].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')