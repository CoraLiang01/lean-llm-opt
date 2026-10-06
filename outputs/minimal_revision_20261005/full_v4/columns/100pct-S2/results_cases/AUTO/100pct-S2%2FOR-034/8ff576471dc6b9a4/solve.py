import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',')
items = df['item'].astype(int).tolist()
value_dict = dict(zip(df['item'].astype(int), df['value'].astype(float)))
weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(float)))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item keys between index set and parameter dictionaries.')
m = gp.Model('KnapsackDisplay')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in items:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')