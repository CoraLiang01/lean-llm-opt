import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',')
items = df['item'].astype(int).tolist()
value_dict = dict(zip(df['item'].astype(int), df['value'].astype(int)))
weight_dict = dict(zip(df['item'].astype(int), df['weight'].astype(int)))
if set(items) != set(value_dict.keys()) or set(items) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item indices between value and weight columns.')
m = gp.Model('KnapsackDisplay')
x = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0
    for i in items:
        if x[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')