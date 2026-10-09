import gurobipy as gp
import pandas as pd
import numpy as np
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv', dtype=str, keep_default_na=False)
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv', dtype=str, keep_default_na=False)
product_ids = products_df['ProductName'].tolist()
warehouse_ids = capacity_df['Warehouse ID'].tolist()
value_dict = {}
weight_dict = {}
for (idx, row) in products_df.iterrows():
    pid = row['ProductName']
    try:
        value_dict[pid] = int(row['Value'])
        weight_dict[pid] = int(row['Weight'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in products.csv for product '{pid}': {e}")
capacity_dict = {}
for (idx, row) in capacity_df.iterrows():
    wid = row['Warehouse ID']
    try:
        capacity_dict[wid] = int(row['Capacity'])
    except Exception as e:
        raise ValueError(f"Invalid numeric value in capacity.csv for warehouse '{wid}': {e}")
if set(product_ids) != set(value_dict.keys()) or set(product_ids) != set(weight_dict.keys()):
    raise ValueError('Mismatch in product identifiers between products.csv and parameter dictionaries.')
if set(warehouse_ids) != set(capacity_dict.keys()):
    raise ValueError('Mismatch in warehouse identifiers between capacity.csv and parameter dictionary.')
m = gp.Model('CarInventoryReplenishment')
x_vars = m.addVars(product_ids, warehouse_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[p] * x_vars[p, w] for p in product_ids for w in warehouse_ids)), gp.GRB.MAXIMIZE)
for w in warehouse_ids:
    m.addConstr(gp.quicksum((weight_dict[p] * x_vars[p, w] for p in product_ids)) <= capacity_dict[w], name=f'cap_{w}')
m.optimize()