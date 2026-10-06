import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv'
df_capacity = pd.read_csv(capacity_path, sep=',')
if df_capacity.isnull().any().any():
    raise ValueError('Missing values detected in capacity.csv')
if not set(['ShelfID', 'Capacity']).issubset(df_capacity.columns):
    raise ValueError('capacity.csv missing required columns')
df_products = pd.read_csv(products_path, sep=',')
if df_products.isnull().any().any():
    raise ValueError('Missing values detected in products.csv')
if not set(['ProductName', 'Value', 'Weight']).issubset(df_products.columns):
    raise ValueError('products.csv missing required columns')
shelves = df_capacity['ShelfID'].astype(int).unique().tolist()
products = df_products['ProductName'].astype(int).unique().tolist()
capacity = df_capacity.set_index('ShelfID')['Capacity'].astype(int).to_dict()
value = df_products.set_index('ProductName')['Value'].astype(int).to_dict()
weight = df_products.set_index('ProductName')['Weight'].astype(int).to_dict()
if set(shelves) != set(capacity.keys()):
    raise ValueError('Mismatch between shelves and capacity keys')
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch between products and value/weight keys')
keys = [(i, j) for i in shelves for j in products]
m = gp.Model('BigMartShelfAllocation')
x = m.addVars(keys, vtype=gp.GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((value[j] * x[i, j] for i in shelves for j in products)), gp.GRB.MAXIMIZE)
for i in shelves:
    m.addConstr(gp.quicksum((weight[j] * x[i, j] for j in products)) <= capacity[i], name=f'cap_{i}')
m.Params.MIPGap = 0.0001
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'ObjVal: {m.objVal}')
    for i in shelves:
        for j in products:
            var = x[i, j]
            print(f'{var.VarName} {var.X}')
else:
    print(f'Solver status: {m.status}')