import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Missing required columns: 'item', 'value', or 'weight' in value.csv")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value_param = dict(zip(df['item'], df['value']))
weight_param = dict(zip(df['item'], df['weight']))
if set(item_ids) != set(value_param.keys()) or set(item_ids) != set(weight_param.keys()):
    raise ValueError('Mismatch in item index coverage for value or weight parameters.')
m = gp.Model('KnapsackDisplay')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in item_ids:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')