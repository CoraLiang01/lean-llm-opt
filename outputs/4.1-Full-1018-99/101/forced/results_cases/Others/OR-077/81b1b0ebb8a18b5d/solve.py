import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'item', 'value', 'weight'}.issubset(df.columns):
    raise KeyError('Missing required columns in value.csv')
df['item'] = df['item'].str.strip()
df['value'] = df['value'].str.strip()
df['weight'] = df['weight'].str.strip()
try:
    df['item'] = df['item'].astype(int)
    df['value'] = df['value'].astype(int)
    df['weight'] = df['weight'].astype(int)
except Exception as e:
    raise ValueError(f'Failed to convert item/value/weight columns to int: {e}')
item_ids = df['item'].tolist()
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
    total_weight = sum((weight_dict[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight}')
else:
    print(f'No optimal solution found. Status: {m.status}')