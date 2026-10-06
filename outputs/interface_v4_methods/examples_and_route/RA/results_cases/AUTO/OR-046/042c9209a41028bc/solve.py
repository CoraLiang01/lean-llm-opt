import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
capacity_df = pd.read_csv(capacity_path, sep=',')
required_product_cols = {'ProductName', 'Weight', 'Value'}
if not required_product_cols.issubset(products_df.columns):
    missing = required_product_cols - set(products_df.columns)
    raise KeyError(f'Missing columns in products.csv: {missing}')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Missing 'Capacity' column in capacity.csv")
product_ids = products_df['ProductName'].astype(str).tolist()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row.')
capacity = int(capacity_df.iloc[0]['Capacity'])

def solve_supermarket_knapsack(product_ids, weight, value, capacity):
    m = gp.Model('SupermarketStockReplenishment')
    x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
    m.optimize()
    return m
m = solve_supermarket_knapsack(product_ids, weight, value, capacity)