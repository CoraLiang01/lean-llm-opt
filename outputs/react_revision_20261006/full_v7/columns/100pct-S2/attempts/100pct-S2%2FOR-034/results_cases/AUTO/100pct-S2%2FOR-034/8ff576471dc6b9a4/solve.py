import gurobipy as gp
import pandas as pd
import numpy as np
import re
import math
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns:
    raise KeyError("Required column 'item' not found in CSV.")
item_ids = df['item'].astype(str).tolist()
if 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Required columns 'value' and/or 'weight' not found in CSV.")
value_param = {}
weight_param = {}
for (idx, row) in df.iterrows():
    item = str(row['item'])
    try:
        value = int(row['value'])
        weight = int(row['weight'])
    except Exception as e:
        raise ValueError(f'Non-integer value or weight for item {item}: {e}')
    value_param[item] = value
    weight_param[item] = weight
if set(item_ids) != set(value_param.keys()) or set(item_ids) != set(weight_param.keys()):
    raise ValueError('Mismatch between item IDs and value/weight parameter keys.')
m = gp.Model('KnapsackDisplay')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[item] * x_vars[item] for item in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[item] * x_vars[item] for item in item_ids)) <= 15, name='weight_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for item in item_ids:
        print(f'{x_vars[item].VarName} {x_vars[item].X}')
else:
    print(f'Solver status: {m.status}')