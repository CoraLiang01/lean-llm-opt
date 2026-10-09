import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    total_capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f"Could not convert capacity value to int: {capacity_df.loc[0, 'Capacity']}") from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
required_columns = ['ProductName', 'Value', 'Weight']
for col in required_columns:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(int)
except Exception as e:
    raise ValueError("Could not convert 'Value' or 'Weight' columns to int in products.csv") from e
product_ids = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(value_dict.keys()) != set(product_ids) or set(weight_dict.keys()) != set(product_ids):
    raise ValueError('Mismatch in product identifiers between value/weight and product list.')
m = gp.Model('PharmacyInventoryKnapsack')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= total_capacity, name='TotalWeightCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Order Plan ---')
    for i in product_ids:
        qty = x_vars[i].X
        if qty > 0.5:
            print(f'  {i}: {int(round(qty))} units (Value/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
    total_weight = sum((weight_dict[i] * x_vars[i].X for i in product_ids))
    print(f'Total weight used: {total_weight:.0f} / {total_capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')