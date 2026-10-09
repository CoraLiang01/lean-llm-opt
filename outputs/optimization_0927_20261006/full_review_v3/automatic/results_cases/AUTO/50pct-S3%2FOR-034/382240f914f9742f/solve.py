import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/50pct/S3/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_cols = ['item', 'value', 'weight']
for col in required_cols:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
items = df['item'].tolist()
value_param = dict(zip(df['item'], df['value']))
weight_param = dict(zip(df['item'], df['weight']))
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_param[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_param[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.0f}')
    print('--- Selected Items ---')
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_param[i]}, weight={weight_param[i]}')
    total_weight = sum((weight_param[i] for i in items if x_vars[i].X > 0.5))
    print(f'Total weight used: {total_weight} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')