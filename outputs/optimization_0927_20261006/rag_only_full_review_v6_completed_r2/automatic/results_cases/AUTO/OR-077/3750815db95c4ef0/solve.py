import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'item', 'value', 'weight'}.issubset(df.columns):
    raise ValueError('Missing required columns in value.csv')
df['item'] = df['item'].str.strip()
df['item'] = df['item'].astype(int)
df['value'] = df['value'].str.strip().astype(int)
df['weight'] = df['weight'].str.strip().astype(int)
item_ids = df['item'].tolist()
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
if set(value_dict.keys()) != set(item_ids) or set(weight_dict.keys()) != set(item_ids):
    raise ValueError('Mismatch in item identifiers for value or weight.')
m = Model('shopping_centre_knapsack')
x_vars = m.addVars(item_ids, vtype=GRB.BINARY, lb=0, ub=1, name='')
m.setObjective(quicksum((value_dict[i] * x_vars[i] for i in item_ids)), GRB.MAXIMIZE)
m.addConstr(quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()