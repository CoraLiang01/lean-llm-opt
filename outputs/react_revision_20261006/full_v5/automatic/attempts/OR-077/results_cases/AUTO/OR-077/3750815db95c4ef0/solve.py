import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
if df['item'].isnull().any():
    raise ValueError('Missing item identifiers in value.csv')
if df['item'].duplicated().any():
    raise ValueError('Duplicate item identifiers in value.csv')
items = df['item'].astype(int).tolist()
values = df.set_index('item')['value'].astype(float).to_dict()
weights = df.set_index('item')['weight'].astype(float).to_dict()
for i in items:
    if i not in values or i not in weights:
        raise ValueError(f'Missing value or weight for item {i}')
m = gp.Model('Knapsack')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((values[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weights[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in items:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')