import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', dtype=str, keep_default_na=False)
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value = dict(zip(df['item'], df['value']))
weight = dict(zip(df['item'], df['weight']))
if set(item_ids) != set(value.keys()) or set(item_ids) != set(weight.keys()):
    raise ValueError('Mismatch in item indices between value and weight parameters.')
m = Model('shopping_centre_knapsack')
x_vars = m.addVars(item_ids, vtype=GRB.BINARY, name='')
m.setObjective(quicksum((value[i] * x_vars[i] for i in item_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((weight[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()