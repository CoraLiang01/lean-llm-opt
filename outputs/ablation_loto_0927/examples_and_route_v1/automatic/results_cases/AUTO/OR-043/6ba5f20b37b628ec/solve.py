import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
for col in ['ProductName', 'Value', 'Weight']:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
capacity_df = pd.read_csv(capacity_path, sep=',')
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row')
capacity = int(capacity_df['Capacity'].iloc[0])
product_ids = products_df['ProductName'].astype(str).tolist()
value_dict = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight_dict = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between Value and Weight columns')
m = gp.Model('PharmacyDrugOrder')
x = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Drug Order Plan ---')
    for i in product_ids:
        xi = x[i].X
        if xi > 0.5:
            print(f'{i}: {int(round(xi))} units (Value/unit: {value_dict[i]}, Weight/unit: {weight_dict[i]})')
    total_weight = sum((weight_dict[i] * x[i].X for i in product_ids))
    print(f'Total stock space used: {int(round(total_weight))} / {capacity}')
else:
    print(f'No optimal solution found. Status: {m.status}')