import gurobipy as gp
import pandas as pd
import numpy as np
value_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(value_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['item', 'value', 'weight']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in value.csv")
df['item'] = df['item'].astype(int)
df['value'] = df['value'].astype(int)
df['weight'] = df['weight'].astype(int)
item_ids = df['item'].tolist()
value_dict = dict(zip(df['item'], df['value']))
weight_dict = dict(zip(df['item'], df['weight']))
if set(value_dict.keys()) != set(item_ids) or set(weight_dict.keys()) != set(item_ids):
    raise ValueError('Mismatch in item keys for value or weight columns.')
m = gp.Model('Knapsack')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= 15, name='weight_limit')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in item_ids:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.status}')