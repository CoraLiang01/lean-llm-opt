import gurobipy as gp
import pandas as pd
import numpy as np
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',')
products_df['ProductName'] = products_df['ProductName'].astype(str).str.strip()
product_ids = products_df['ProductName'].tolist()
value_dict = dict(zip(products_df['ProductName'], products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity_df = pd.read_csv(capacity_path, sep=',')
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].astype(str).str.strip()
warehouse_ids = capacity_df['Warehouse ID'].tolist()
capacity_dict = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity']))
if len(product_ids) == 0 or len(warehouse_ids) == 0:
    raise ValueError('No products or warehouses found in the input data.')
for pid in product_ids:
    if pid not in value_dict or pid not in weight_dict:
        raise ValueError(f"Missing value or weight for product '{pid}'.")
for wid in warehouse_ids:
    if wid not in capacity_dict:
        raise ValueError(f"Missing capacity for warehouse '{wid}'.")

def solve_inventory_placement(product_ids, warehouse_ids, value_dict, weight_dict, capacity_dict):
    m = gp.Model('CarInventoryPlacement')
    x = m.addVars(product_ids, warehouse_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[i] * x[i, w] for i in product_ids for w in warehouse_ids)), gp.GRB.MAXIMIZE)
    for w in warehouse_ids:
        m.addConstr(gp.quicksum((weight_dict[i] * x[i, w] for i in product_ids)) <= capacity_dict[w], name=f'cap_{w}')
    m.optimize()
    return m
m = solve_inventory_placement(product_ids, warehouse_ids, value_dict, weight_dict, capacity_dict)