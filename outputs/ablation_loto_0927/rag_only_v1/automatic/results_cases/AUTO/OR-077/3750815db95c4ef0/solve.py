import pandas as pd
import numpy as np
from gurobipy import Model, GRB
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
value_df['item'] = value_df['item'].astype(int)
items = value_df['item'].tolist()
value_dict = dict(zip(value_df['item'], value_df['value']))
weight_dict = dict(zip(value_df['item'], value_df['weight']))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item indices between value and weight data.')
m = Model('shopping_centre_knapsack')
x = m.addVars(items, vtype=GRB.BINARY, name='')
m.setObjective(sum((value_dict[i] * x[i] for i in items)), GRB.MAXIMIZE)
m.addConstr(sum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()