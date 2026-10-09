import pandas as pd
import numpy as np
from gurobipy import Model, GRB, quicksum
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/capacity.csv', sep=',', dtype=str, keep_default_na=False)
if 'Capacity' not in capacity_df.columns:
    raise ValueError("Missing 'Capacity' column in capacity.csv")
if capacity_df.shape[0] != 1:
    raise ValueError('Expected exactly one row in capacity.csv')
try:
    capacity = int(capacity_df.loc[0, 'Capacity'])
except Exception as e:
    raise ValueError(f"Invalid capacity value: {capacity_df.loc[0, 'Capacity']}") from e
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA8/PharmaSalesData1/products.csv', sep=',', dtype=str, keep_default_na=False)
required_columns = ['ProductName', 'Value', 'Weight']
for col in required_columns:
    if col not in products_df.columns:
        raise ValueError(f"Missing '{col}' column in products.csv")
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(str).str.strip()
products_df['Weight'] = products_df['Weight'].astype(str).str.strip()

def safe_int(val, col, idx):
    try:
        return int(val)
    except Exception:
        raise ValueError(f"Invalid {col} value '{val}' for product '{products_df.loc[idx, 'ProductName']}'")
products_df['Value'] = [safe_int(v, 'Value', idx) for (idx, v) in enumerate(products_df['Value'])]
products_df['Weight'] = [safe_int(w, 'Weight', idx) for (idx, w) in enumerate(products_df['Weight'])]
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(product_names, products_df['Value']))
weight_dict = dict(zip(product_names, products_df['Weight']))
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in product identifiers between Value and Weight columns.')
m = Model('PharmacyRestock')
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(quicksum((value_dict[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstr(quicksum((weight_dict[i] * x_vars[i] for i in product_names)) <= capacity, name='capacity')
m.optimize()