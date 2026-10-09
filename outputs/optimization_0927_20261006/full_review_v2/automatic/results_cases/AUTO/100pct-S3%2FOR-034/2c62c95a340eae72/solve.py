import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def norm_col(col):
    return col.strip().casefold()
colmap = {norm_col(c): c for c in df.columns}
item_col = colmap['item']
value_col = colmap['value']
weight_col = colmap['weight']
item_ids = df[item_col].astype(str).tolist()
value_param = df.set_index(item_col)[value_col].astype(int).to_dict()
weight_param = df.set_index(item_col)[weight_col].astype(int).to_dict()
m = gp.Model('KnapsackDisplay')
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