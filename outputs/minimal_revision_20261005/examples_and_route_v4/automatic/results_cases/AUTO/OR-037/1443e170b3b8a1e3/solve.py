import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA3/CarSales2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'products.csv is missing columns: {missing}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
product_keys = list(products_df['ProductName'])
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(product_keys) != set(value_dict.keys()) or set(product_keys) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product keys and parameter dictionaries.')

def solve_problem(product_keys, value_dict, weight_dict, capacity):
    m = gp.Model('CarSalesInventory')
    x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_keys)) <= capacity, name='cap')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(product_keys, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')