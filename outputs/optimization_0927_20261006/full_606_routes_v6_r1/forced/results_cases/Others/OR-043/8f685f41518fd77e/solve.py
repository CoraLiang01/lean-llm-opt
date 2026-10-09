import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must have exactly one row')
try:
    total_capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f"Could not convert 'Capacity' value to int: {e}")
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', sep=',', dtype=str, keep_default_na=False)
required_cols = ['ProductName', 'Value', 'Weight']
for col in required_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(int)
except Exception as e:
    raise ValueError(f"Could not convert 'Value' or 'Weight' to int: {e}")
product_ids = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between index set and parameter dictionaries.')
m = gp.Model('PharmacyDrugOrder')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[prod] * x_vars[prod] for prod in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[prod] * x_vars[prod] for prod in product_ids)) <= total_capacity, name='TotalCapacity')
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Drug Order Plan ---')
    for prod in product_ids:
        qty = x_vars[prod].X
        if qty > 0.5:
            print(f'{prod}: {int(round(qty))} units (Value/unit: {value_dict[prod]}, Weight/unit: {weight_dict[prod]})')
else:
    print(f'No optimal solution found. Status: {m.status}')