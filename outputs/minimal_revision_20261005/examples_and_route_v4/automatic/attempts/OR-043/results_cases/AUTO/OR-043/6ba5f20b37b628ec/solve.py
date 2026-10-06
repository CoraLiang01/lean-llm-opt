import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    missing = required_cols - set(products_df.columns)
    raise ValueError(f'Missing columns in products.csv: {missing}')
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
capacity = int(capacity_df['Capacity'].iloc[0])
product_keys = products_df['ProductName'].astype(str).tolist()
value_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Value'].astype(int)))
weight_dict = dict(zip(products_df['ProductName'].astype(str), products_df['Weight'].astype(int)))
if set(product_keys) != set(value_dict.keys()) or set(product_keys) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between index set and parameter dictionaries.')
m = gp.Model('PharmacyDrugOrder')
m.Params.MIPGap = 0.0001
x = m.addVars(product_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_keys)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_keys)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in product_keys:
        print(f'{x[i].VarName} {x[i].X}')
else:
    print(f'Solver status: {m.status}')