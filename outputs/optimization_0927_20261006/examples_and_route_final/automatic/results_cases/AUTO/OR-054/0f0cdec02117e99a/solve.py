import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', dtype=str, keep_default_na=False)
shelf_ids = capacity_df['ShelfID'].astype(int).tolist()
product_ids = products_df['ProductName'].astype(int).tolist()
capacity_dict = dict(zip(capacity_df['ShelfID'].astype(int), capacity_df['Capacity'].astype(int)))
value_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(int), products_df['Weight'].astype(int)))
if set(shelf_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch between shelf_ids and capacity_dict keys.')
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch between product_ids and value/weight dict keys.')
m = gp.Model('BigMart_MultiKnapsack')
x_vars = m.addVars(shelf_ids, product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[j] * x_vars[i, j] for i in shelf_ids for j in product_ids)), gp.GRB.MAXIMIZE)
for i in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[j] * x_vars[i, j] for j in product_ids)) <= capacity_dict[i], name=f'shelf_cap_{i}')
m.optimize()