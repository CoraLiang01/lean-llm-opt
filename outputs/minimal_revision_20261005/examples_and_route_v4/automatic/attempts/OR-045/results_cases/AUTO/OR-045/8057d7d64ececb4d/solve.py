import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA11/SupermarketSalesData1/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row and a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Weight', 'Value'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'products.csv is missing columns: {missing}')
product_keys = list(products_df['ProductName'])
if len(set(product_keys)) != len(product_keys):
    raise ValueError('ProductName column in products.csv must have unique values.')
weight = {}
value = {}
for (idx, row) in products_df.iterrows():
    pname = row['ProductName']
    try:
        w = int(row['Weight'])
        v = int(row['Value'])
    except Exception:
        raise ValueError(f"Non-integer Weight or Value for product '{pname}'.")
    weight[pname] = w
    value[pname] = v
if set(weight.keys()) != set(product_keys) or set(value.keys()) != set(product_keys):
    raise ValueError('Mismatch in product keys for weights/values.')

def solve_supermarket_knapsack(product_keys, weight, value, capacity):
    m = gp.Model('SupermarketKnapsack')
    m.Params.MIPGap = 0.0001
    x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
    m.addConstr(gp.quicksum((weight[i] * x[i] for i in product_keys)) <= capacity, name='cap')
    m.optimize()
    return m
m = solve_supermarket_knapsack(product_keys, weight, value, capacity)
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')