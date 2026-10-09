import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Weight', 'Value'}.issubset(products_df.columns):
    raise ValueError('products.csv must contain columns: ProductName, Weight, Value')
products_df['Weight'] = products_df['Weight'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must contain column: Capacity')
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
capacity = int(capacity_df.iloc[0]['Capacity'])
product_keys = products_df['ProductName'].tolist()
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
value = dict(zip(products_df['ProductName'], products_df['Value']))
if set(weight.keys()) != set(product_keys) or set(value.keys()) != set(product_keys):
    raise ValueError('Mismatch in product keys between weight/value and product list.')

def solve_problem(product_keys, weight, value, capacity):
    m = gp.Model('SupermarketRestock')
    quantity_vars = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * quantity_vars[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * quantity_vars[i] for i in product_keys)) <= capacity, name='capacity')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem(product_keys, weight, value, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')