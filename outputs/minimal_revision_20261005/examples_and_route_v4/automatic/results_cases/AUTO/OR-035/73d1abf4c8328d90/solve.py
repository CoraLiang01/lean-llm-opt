import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
if products_df.shape[0] == 0:
    raise ValueError('products.csv is empty.')
capacity_df = pd.read_csv(capacity_path, sep=',')
if capacity_df.shape[0] == 0 or 'Capacity' not in capacity_df.columns:
    raise ValueError("capacity.csv is empty or missing 'Capacity' column.")
product_keys = products_df['ProductName'].astype(str).tolist()
if len(set(product_keys)) != len(product_keys):
    raise ValueError('Duplicate ProductName entries found in products.csv.')
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
for k in product_keys:
    if k not in value_dict or k not in weight_dict:
        raise ValueError(f"Missing Value or Weight for product '{k}'.")
if capacity_df.shape[0] != 1:
    raise ValueError('capacity.csv must have exactly one row.')
capacity = int(capacity_df.iloc[0]['Capacity'])

def solve_bakery_knapsack(product_keys, value_dict, weight_dict, capacity):
    m = gp.Model('BakeryKnapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_keys)) <= capacity, name='storage')
    m.optimize()
    return m
m = solve_bakery_knapsack(product_keys, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')