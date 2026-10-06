import gurobipy as gp
import pandas as pd
import numpy as np
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv', sep=',')
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv', sep=',')
if 'Capacity' not in capacity_df.columns or capacity_df.shape[0] != 1:
    raise ValueError("capacity.csv must have exactly one row with a 'Capacity' column.")
capacity = int(capacity_df['Capacity'].iloc[0])
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must contain columns: {required_cols}')
products_df['ProductName'] = products_df['ProductName'].astype(str)
products = list(products_df['ProductName'])
value = products_df.set_index('ProductName')['Value'].astype(int).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(int).to_dict()
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch in product identifiers between columns.')
m = gp.Model('BakeryBreadOrder')
x = m.addVars(products, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[i] * x[i] for i in products)), gp.GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[i] * x[i] for i in products)) <= capacity, name='storage_capacity')
m.optimize()