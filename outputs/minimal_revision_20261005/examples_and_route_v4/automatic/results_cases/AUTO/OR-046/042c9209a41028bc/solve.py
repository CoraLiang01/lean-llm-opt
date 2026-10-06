import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA12/SupermarketSalesData2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df.loc[0, 'Capacity'])
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'products.csv is missing columns: {missing}')
product_keys = list(products_df['ProductName'])
if len(set(product_keys)) != len(product_keys):
    raise ValueError('ProductName column in products.csv must have unique values.')
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
if set(weight.keys()) != set(product_keys) or set(value.keys()) != set(product_keys):
    raise ValueError('Mismatch in product keys for weights/values.')
m = gp.Model('SupermarketStockReplenishment')
x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_keys)) <= capacity, name='stock_capacity')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in product_keys:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')