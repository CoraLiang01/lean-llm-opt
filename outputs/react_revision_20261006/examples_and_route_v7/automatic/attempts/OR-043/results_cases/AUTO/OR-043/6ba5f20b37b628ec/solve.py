import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'Missing columns in products.csv: {missing}')
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row for total capacity.')
product_keys = products_df['ProductName'].tolist()
if len(set(product_keys)) != len(product_keys):
    raise ValueError('Duplicate ProductName entries found in products.csv.')
try:
    value_dict = dict(zip(product_keys, products_df['Value'].astype(int)))
    weight_dict = dict(zip(product_keys, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to int: {e}')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')

def solve_pharma_inventory(product_keys, value_dict, weight_dict, capacity):
    m = gp.Model('PharmaInventory')
    quantity_vars = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * quantity_vars[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight_dict[i] * quantity_vars[i] for i in product_keys)) <= capacity, name='stock_cap')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_pharma_inventory(product_keys, value_dict, weight_dict, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')