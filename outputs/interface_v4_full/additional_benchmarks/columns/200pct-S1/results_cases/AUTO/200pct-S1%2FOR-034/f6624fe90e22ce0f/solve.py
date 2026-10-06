import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
if df['item'].isnull().any():
    raise ValueError('Missing item identifiers in value.csv')
items = df['item'].astype(int).tolist()
if df['value'].isnull().any():
    raise ValueError('Missing value data for some items in value.csv')
if df['weight'].isnull().any():
    raise ValueError('Missing weight data for some items in value.csv')
value = dict(zip(df['item'].astype(int), df['value'].astype(int)))
weight = dict(zip(df['item'].astype(int), df['weight'].astype(int)))
if set(items) != set(value.keys()) or set(items) != set(weight.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')
capacity = 15
m = gp.Model('KnapsackSelection')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in items)) <= capacity, name='weight_limit')
m.optimize()