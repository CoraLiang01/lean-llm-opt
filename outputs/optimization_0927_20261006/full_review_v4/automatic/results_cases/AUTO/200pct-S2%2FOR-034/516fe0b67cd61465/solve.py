import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/200pct/S2/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
df['item'] = df['item'].astype(int)
if df['item'].duplicated().any():
    raise ValueError('Duplicate item IDs found in value.csv')
items = df['item'].tolist()
try:
    value_dict = pd.Series(df['value'].astype(int).values, index=df['item']).to_dict()
    weight_dict = pd.Series(df['weight'].astype(int).values, index=df['item']).to_dict()
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to int: {e}')
m = gp.Model('KnapsackDisplaySelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in items if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')