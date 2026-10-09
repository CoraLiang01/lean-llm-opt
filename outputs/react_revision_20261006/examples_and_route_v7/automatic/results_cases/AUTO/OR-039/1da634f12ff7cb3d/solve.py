import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA5/NewCarSalesInNorway2/products.csv'
capacity_df = pd.read_csv(capacity_path, sep=',', dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, sep=',', dtype=str, keep_default_na=False)
warehouse_ids = capacity_df['Warehouse ID'].str.strip().tolist()
product_ids = products_df['ProductName'].str.strip().tolist()
capacity_df['Capacity'] = capacity_df['Capacity'].astype(int)
products_df['Value'] = products_df['Value'].astype(int)
products_df['Weight'] = products_df['Weight'].astype(int)
capacity_dict = dict(zip(capacity_df['Warehouse ID'].str.strip(), capacity_df['Capacity']))
value_dict = dict(zip(products_df['ProductName'].str.strip(), products_df['Value']))
weight_dict = dict(zip(products_df['ProductName'].str.strip(), products_df['Weight']))
if len(set(warehouse_ids)) != len(warehouse_ids):
    raise ValueError('Duplicate warehouse IDs found in capacity.csv')
if len(set(product_ids)) != len(product_ids):
    raise ValueError('Duplicate product names found in products.csv')
if set(capacity_dict.keys()) != set(warehouse_ids):
    raise ValueError('Mismatch in warehouse IDs between DataFrame and dict')
if set(value_dict.keys()) != set(product_ids) or set(weight_dict.keys()) != set(product_ids):
    raise ValueError('Mismatch in product IDs between DataFrame and dict')
decision_keys = [(w, p) for w in warehouse_ids for p in product_ids]
m = gp.Model('NewCarSalesInventory')
x_vars = m.addVars(decision_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value_dict[p] * x_vars[w, p] for w in warehouse_ids for p in product_ids)), gp.GRB.MAXIMIZE)
for w in warehouse_ids:
    m.addConstr(gp.quicksum((weight_dict[p] * x_vars[w, p] for p in product_ids)) <= capacity_dict[w], name=f'cap_{w}')
m.setParam('MIPGap', 0.0001)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for w in warehouse_ids:
        for p in product_ids:
            var = x_vars[w, p]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')