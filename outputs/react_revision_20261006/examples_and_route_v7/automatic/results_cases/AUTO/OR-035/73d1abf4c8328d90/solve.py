import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
required_product_cols = {'ProductName', 'Value', 'Weight'}
if not required_product_cols.issubset(products_df.columns):
    missing = required_product_cols - set(products_df.columns)
    raise ValueError(f'Missing columns in products.csv: {missing}')
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
product_ids = products_df['ProductName'].astype(str).tolist()
try:
    value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise ValueError(f"Missing Value or Weight for product '{pid}'")

def solve_bakery_knapsack(product_ids, value_dict, weight_dict, capacity):
    m = gp.Model('BakeryKnapsack')
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= capacity, name='storage')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_bakery_knapsack(product_ids, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')