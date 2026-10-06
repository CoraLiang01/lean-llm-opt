import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA2/CarSales1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'products.csv is missing required columns: {missing}')
product_keys = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(product_keys) != set(value_dict.keys()) or set(product_keys) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product keys between index set and parameter dictionaries.')

def solve_inventory_replenishment(product_keys, value_dict, weight_dict, capacity):
    m = gp.Model('CarSalesInventory')
    x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_keys)) <= capacity, name='capacity')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_inventory_replenishment(product_keys, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')