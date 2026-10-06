import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete 20260928/200pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
items = df['item'].astype(int).tolist()
value_dict = pd.Series(df['value'].values, index=df['item'].astype(int)).to_dict()
weight_dict = pd.Series(df['weight'].values, index=df['item'].astype(int)).to_dict()
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch between items and value/weight keys.')
m = gp.Model('KnapsackDisplaySelection')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()