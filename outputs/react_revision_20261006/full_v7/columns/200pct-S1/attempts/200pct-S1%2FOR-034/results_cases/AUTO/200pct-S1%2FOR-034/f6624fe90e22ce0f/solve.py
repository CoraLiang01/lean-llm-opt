import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if 'item' not in df.columns or 'value' not in df.columns or 'weight' not in df.columns:
    raise KeyError("Missing required columns: 'item', 'value', or 'weight' in value.csv")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
if set(item_ids) != set(value_dict.keys()) or set(item_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item indices and parameter keys.')

def solve_knapsack(item_ids, value_dict, weight_dict, weight_limit):
    m = gp.Model('knapsack')
    x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= weight_limit, name='weight_limit')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_knapsack(item_ids, value_dict, weight_dict, weight_limit=15)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')