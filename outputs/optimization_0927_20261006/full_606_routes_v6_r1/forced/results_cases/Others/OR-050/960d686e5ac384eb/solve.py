import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise KeyError("capacity.csv must contain 'ShelfID' and 'Capacity' columns.")
capacity_df['ShelfID'] = capacity_df['ShelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
try:
    capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(int)
    capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting ShelfID or Capacity to numeric: {e}')
display_ids = list(capacity_df['ShelfID'])
display_capacities = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if 'ProductName' not in products_df.columns or 'Value' not in products_df.columns or 'Weight' not in products_df.columns:
    raise KeyError("products.csv must contain 'ProductName', 'Value', and 'Weight' columns.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
try:
    products_df['Value'] = products_df['Value'].astype(int)
    products_df['Weight'] = products_df['Weight'].astype(float)
except Exception as e:
    raise ValueError(f'Error converting Value or Weight to numeric: {e}')
product_names = list(products_df['ProductName'])
product_values = dict(zip(products_df['ProductName'], products_df['Value']))
product_weights = dict(zip(products_df['ProductName'], products_df['Weight']))
if len(product_names) == 0:
    raise ValueError('No products found in products.csv.')
first_product = product_names[0]
m = gp.Model('RetailDisplayAllocation')
x_vars = m.addVars(display_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((product_values[prod] * x_vars[disp, prod] for disp in display_ids for prod in product_names)), gp.GRB.MAXIMIZE)
for disp in display_ids:
    m.addConstr(gp.quicksum((product_weights[prod] * x_vars[disp, prod] for prod in product_names)) <= display_capacities[disp], name=f'capacity_{disp}')
m.addConstr(gp.quicksum((x_vars[disp, first_product] for disp in display_ids)) >= 5, name='min_first_product')
m.optimize()