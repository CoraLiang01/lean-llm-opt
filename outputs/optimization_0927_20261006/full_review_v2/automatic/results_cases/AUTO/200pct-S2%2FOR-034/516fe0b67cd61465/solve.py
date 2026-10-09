import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
col_map = {norm_col(col): col for col in df.columns}
required_cols = ['item', 'value', 'weight']
for rc in required_cols:
    if rc not in [norm_col(c) for c in df.columns]:
        raise KeyError(f"Required column '{rc}' not found in CSV.")
item_col = col_map['item']
value_col = col_map['value']
weight_col = col_map['weight']
df_items = df.copy()
df_items['item_id'] = df_items[item_col].astype(int)
item_ids = df_items['item_id'].tolist()
value_dict = dict(zip(df_items['item_id'], df_items[value_col].astype(int)))
weight_dict = dict(zip(df_items['item_id'], df_items[weight_col].astype(int)))
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')