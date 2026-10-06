import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA22/BoatSales2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity['DisplayID'].isnull().any() or df_capacity['Capacity'].isnull().any():
    raise ValueError('Missing DisplayID or Capacity in capacity.csv')
display_ids = df_capacity['DisplayID'].astype(int).tolist()
capacity_dict = dict(zip(df_capacity['DisplayID'].astype(int), df_capacity['Capacity'].astype(int)))
df_products = pd.read_csv(products_path, sep=',')
if df_products['ProductName'].isnull().any() or df_products['Value'].isnull().any() or df_products['Weight'].isnull().any():
    raise ValueError('Missing ProductName, Value, or Weight in products.csv')
product_names = df_products['ProductName'].astype(str).tolist()
value_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Value'].astype(int)))
weight_dict = dict(zip(df_products['ProductName'].astype(str), df_products['Weight'].astype(int)))
if set(capacity_dict.keys()) != set(display_ids):
    raise ValueError('Mismatch in display IDs in capacity.csv')
if set(value_dict.keys()) != set(product_names) or set(weight_dict.keys()) != set(product_names):
    raise ValueError('Mismatch in product names in products.csv')
keys = [(i, j) for i in display_ids for j in product_names]

def solve_boat_display_assignment():
    m = gp.Model('BoatDisplayAssignment')
    x = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
    m.setObjective(gp.quicksum((value_dict[j] * x[i, j] for (i, j) in keys)), gp.GRB.MAXIMIZE)
    for i in display_ids:
        m.addConstr(gp.quicksum((weight_dict[j] * x[i, j] for j in product_names)) <= capacity_dict[i], name=f'cap_{i}')
    m.setParam('MIPGap', 0.0001)
    m.optimize()
    return m
m = solve_boat_display_assignment()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for v in m.getVars():
        print(f'{v.VarName} {v.X}')
else:
    print(f'Solver status: {m.status}')