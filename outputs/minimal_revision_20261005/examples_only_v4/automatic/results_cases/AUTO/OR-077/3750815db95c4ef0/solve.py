import gurobipy as gp
import pandas as pd
import numpy as np
value_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv', sep=',')
items = value_df['item'].astype(int).tolist()
values = dict(zip(value_df['item'].astype(int), value_df['value'].astype(float)))
weights = dict(zip(value_df['item'].astype(int), value_df['weight'].astype(float)))
if set(values.keys()) != set(items) or set(weights.keys()) != set(items):
    raise ValueError('Mismatch in item keys between value and weight columns.')
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