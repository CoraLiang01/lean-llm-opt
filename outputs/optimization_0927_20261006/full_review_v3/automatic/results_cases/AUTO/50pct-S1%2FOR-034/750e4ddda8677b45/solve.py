import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
if not {'item', 'value', 'weight'}.issubset(df.columns):
    raise KeyError('Missing required columns in value.csv')
df['item'] = df['item'].astype(str).str.strip()
df['value'] = df['value'].astype(float)
df['weight'] = df['weight'].astype(float)
item_ids = df['item'].tolist()
value_param = dict(zip(df['item'], df['value']))
weight_param = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in item_ids)) <= 15, name='WeightLimit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    for i in item_ids:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]:.0f}, weight={weight_param[i]:.0f}')
    total_weight = sum((weight_param[i] for i in item_ids if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')