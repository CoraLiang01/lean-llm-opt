import gurobipy as gp
import pandas as pd
import numpy as np
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, sep=',')
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
products_df = pd.read_csv(products_path, sep=',')
shelves = capacity_df['ShelfID'].astype(int).tolist()
products = products_df['ProductName'].astype(str).tolist()
if products_df['ProductName'].duplicated().any():
    raise ValueError('Duplicate ProductName found in products.csv; ProductName must be unique.')
value = products_df.set_index('ProductName')['Value'].astype(float).to_dict()
weight = products_df.set_index('ProductName')['Weight'].astype(float).to_dict()
if capacity_df['ShelfID'].duplicated().any():
    raise ValueError('Duplicate ShelfID found in capacity.csv; ShelfID must be unique.')
capacity = capacity_df.set_index('ShelfID')['Capacity'].astype(float).to_dict()
if set(products) != set(value.keys()) or set(products) != set(weight.keys()):
    raise ValueError('Mismatch in products between products.csv and parameter dictionaries.')
if set(shelves) != set(capacity.keys()):
    raise ValueError('Mismatch in shelves between capacity.csv and parameter dictionary.')
x_keys = [(i, j) for i in shelves for j in products]
m = gp.Model('RetailShelfAllocation')
x = m.addVars(x_keys, vtype=gp.GRB.INTEGER, lb=0, name='')
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