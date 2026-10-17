import gurobipy as gp
import pandas as pd
import numpy as np
value_csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_csv_path, sep=',', dtype=str, keep_default_na=False)
df.columns = [col.strip() for col in df.columns]
df['item'] = df['item'].str.strip()
df['value'] = df['value'].str.strip()
df['weight'] = df['weight'].str.strip()
items = df['item'].tolist()
try:
    value_dict = dict(zip(df['item'], df['value'].astype(float)))
    weight_dict = dict(zip(df['item'], df['weight'].astype(float)))
except Exception as e:
    raise ValueError(f'Error converting value or weight columns to float: {e}')
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(items, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in items)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in items)) <= 15, name='weight_limit')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Selected Items ---')
    total_weight = 0.0
    for i in items:
        if x_vars[i].X > 0.5:
            print(f'Item {i}: value={value_dict[i]}, weight={weight_dict[i]}')
            total_weight += weight_dict[i]
    print(f'Total weight used: {total_weight:.2f} / 15')
else:
    print(f'No optimal solution found. Status: {m.status}')