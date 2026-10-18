import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f'Could not convert Capacity value to int: {e}')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', sep=',', dtype=str, keep_default_na=False)
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(products_df['ProductName'], products_df['Value'].astype(int)))
    weight_dict = dict(zip(products_df['ProductName'], products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to int: {e}')
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between Value/Weight and ProductName columns')
m = gp.Model('PharmacyDrugOrderKnapsack')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Drug Order Plan ---')
    for i in product_ids:
        qty = x_vars[i].X
        if qty > 1e-06:
            print(f'{i}: {qty:.0f} units (Value/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')