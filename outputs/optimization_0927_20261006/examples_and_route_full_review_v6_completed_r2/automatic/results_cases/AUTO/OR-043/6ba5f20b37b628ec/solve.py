import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/products.csv', dtype=str, keep_default_na=False)
required_product_cols = ['ProductName', 'Value', 'Weight']
for col in required_product_cols:
    if col not in products_df.columns:
        raise KeyError(f"Column '{col}' not found in products.csv")
product_ids = products_df['ProductName'].tolist()
try:
    value_dict = dict(zip(product_ids, products_df['Value'].astype(int)))
    weight_dict = dict(zip(product_ids, products_df['Weight'].astype(int)))
except Exception as e:
    raise ValueError(f'Error converting Value or Weight columns to int: {e}')
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA9/PharmaSalesData2/capacity.csv', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise KeyError("Column 'Capacity' not found in capacity.csv")
if len(capacity_df) != 1:
    raise ValueError('capacity.csv must contain exactly one row.')
try:
    capacity = int(capacity_df.iloc[0]['Capacity'])
except Exception as e:
    raise ValueError(f'Error converting Capacity to int: {e}')
m = gp.Model('PharmacyDrugOrder')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight_dict[i] * x_vars[i] for i in product_ids)) <= capacity, name='stock_capacity')
m.optimize()