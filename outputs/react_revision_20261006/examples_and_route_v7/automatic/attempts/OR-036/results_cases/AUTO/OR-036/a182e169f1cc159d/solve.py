import gurobipy as gp
import pandas as pd
import numpy as np

def solve_problem():
    products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
    products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
    capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
    capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
    if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
        raise KeyError('products.csv must contain columns: ProductName, Value, Weight')
    product_ids = products_df['ProductName'].tolist()
    if len(set(product_ids)) != len(product_ids):
        raise ValueError('ProductName column in products.csv contains duplicate entries.')
    try:
        value_dict = dict(zip(product_ids, products_df['Value'].astype(int)))
        weight_dict = dict(zip(product_ids, products_df['Weight'].astype(int)))
    except Exception as e:
        raise ValueError(f'Error converting Value or Weight columns to int: {e}')
    if 'Capacity' not in capacity_df.columns:
        raise KeyError('capacity.csv must contain column: Capacity')
    if len(capacity_df) != 1:
        raise ValueError('capacity.csv must contain exactly one row.')
    try:
        capacity = int(capacity_df.iloc[0]['Capacity'])
    except Exception as e:
        raise ValueError(f'Error converting Capacity to int: {e}')
    for pid in product_ids:
        if pid not in value_dict or pid not in weight_dict:
            raise ValueError(f'Missing Value or Weight for product {pid}')
    m = gp.Model('CarSalesInventory')
    m.Params.MIPGap = 0.0001
    x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= capacity, name='capacity')
    m.optimize()
    return m
m = solve_problem()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')