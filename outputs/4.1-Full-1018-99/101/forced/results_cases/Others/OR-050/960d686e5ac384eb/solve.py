import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
if not {'ShelfID', 'Capacity'}.issubset(capacity_df.columns):
    raise KeyError("capacity.csv must contain columns 'ShelfID' and 'Capacity'.")
capacity_df['ShelfID'] = capacity_df['ShelfID'].str.strip()
capacity_df['Capacity'] = capacity_df['Capacity'].str.strip()
shelf_ids = capacity_df['ShelfID'].tolist()
shelf_ids_int = [int(sid) for sid in shelf_ids]
capacity_dict = dict(zip(shelf_ids, capacity_df['Capacity'].astype(float)))
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
if not {'ProductName', 'Value', 'Weight'}.issubset(products_df.columns):
    raise KeyError("products.csv must contain columns 'ProductName', 'Value', and 'Weight'.")
products_df['ProductName'] = products_df['ProductName'].str.strip()
products_df['Value'] = products_df['Value'].str.strip()
products_df['Weight'] = products_df['Weight'].str.strip()
product_names = products_df['ProductName'].tolist()
value_dict = dict(zip(product_names, products_df['Value'].astype(float)))
weight_dict = dict(zip(product_names, products_df['Weight'].astype(float)))
if len(product_names) == 0:
    raise ValueError('No products found in products.csv.')
first_product = product_names[0]
m = gp.Model('RetailDisplayAllocation')
x_vars = m.addVars(shelf_ids, product_names, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[prod] * x_vars[shelf, prod] for shelf in shelf_ids for prod in product_names)), gp.GRB.MAXIMIZE)
for shelf in shelf_ids:
    m.addConstr(gp.quicksum((weight_dict[prod] * x_vars[shelf, prod] for prod in product_names)) <= capacity_dict[shelf], name=f'capacity_{shelf}')
m.addConstr(gp.quicksum((x_vars[shelf, first_product] for shelf in shelf_ids)) >= 5, name='min_first_product')
m.optimize()