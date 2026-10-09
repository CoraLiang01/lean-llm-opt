import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Missing required columns in value.csv: 'item', 'value', or 'weight'.")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value_param = dict(zip(df['item'], df['value']))
weight_param = dict(zip(df['item'], df['weight']))
if set(item_ids) != set(value_param.keys()) or set(item_ids) != set(weight_param.keys()):
    raise ValueError('Mismatch in item index coverage between items and parameters.')
WEIGHT_LIMIT = 15

def solve_knapsack(item_ids, value_param, weight_param, weight_limit):
    m = gp.Model('KnapsackSelection')
    x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= weight_limit, name='weight_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_knapsack(item_ids, value_param, weight_param, WEIGHT_LIMIT)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')