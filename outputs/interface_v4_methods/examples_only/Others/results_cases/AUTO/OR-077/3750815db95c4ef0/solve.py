import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(float)
df['weight'] = df['weight'].astype(float)
items = df['item'].tolist()
value = dict(zip(df['item'], df['value']))
weight = dict(zip(df['item'], df['weight']))
if set(items) != set(value.keys()) or set(items) != set(weight.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')
m = gp.Model('KnapsackSelection')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= 15, name='WeightLimit')
m.optimize()