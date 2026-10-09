import gurobipy as gp
import pandas as pd
import numpy as np
import re

def norm_str(s):
    return str(s).strip()
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].apply(norm_str)
capacity_df['Warehouse ID'] = capacity_df['Warehouse ID'].apply(norm_str)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products = list(products_df['ProductName'])
warehouses = list(capacity_df['Warehouse ID'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
capacity = dict(zip(capacity_df['Warehouse ID'], capacity_df['Capacity']))
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch in product keys between index set and parameter dictionaries.')
if set(warehouses) != set(capacity.keys()):
    raise ValueError('Mismatch in warehouse keys between index set and capacity dictionary.')
m = gp.Model('CarInventoryReplenishment')
x_vars = m.addVars(products, warehouses, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[p, w] for p in products for w in warehouses)), gp.GRB.MAXIMIZE)
for w in warehouses:
    m.addConstr(gp.quicksum((weight[p] * x_vars[p, w] for p in products)) <= capacity[w], name=f'cap_{w}')
m.optimize()