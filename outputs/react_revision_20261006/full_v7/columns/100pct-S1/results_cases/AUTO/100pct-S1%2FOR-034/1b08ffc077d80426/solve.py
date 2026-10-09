import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/redundancy_complete/100pct/S1/Large-scale-or/Others_example/Others3/value.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
item_ids = df['item'].tolist()
if not set(['value', 'weight']).issubset(df.columns):
    raise ValueError("Missing required columns 'value' or 'weight' in value.csv.")
try:
    value_dict = dict(zip(df['item'], df['value'].astype(float)))
    weight_dict = dict(zip(df['item'], df['weight'].astype(float)))
except Exception as e:
    raise ValueError(f"Error converting 'value' or 'weight' columns to float: {e}")
if set(item_ids) != set(value_dict.keys()) or set(item_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in item identifiers between value and weight columns.')
capacity = 15.0
m = gp.Model('KnapsackSelection')
x_vars = m.addVars(item_ids, vtype=gp.GRB.BINARY, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in item_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in item_ids)) <= capacity, name='weight_limit')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for i in item_ids:
        print(f'{x_vars[i].VarName} {x_vars[i].X}')
else:
    print(f'Solver status: {m.Status}')