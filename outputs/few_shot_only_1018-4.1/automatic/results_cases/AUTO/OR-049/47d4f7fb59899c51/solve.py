import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv', dtype=str, keep_default_na=False)
capacity_df['ShelfID'] = capacity_df['ShelfID'].astype(str)
capacity_df['Capacity'] = capacity_df['Capacity'].astype(float)
shelves = list(capacity_df['ShelfID'])
capacity = dict(zip(capacity_df['ShelfID'], capacity_df['Capacity']))
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv', dtype=str, keep_default_na=False)
products_df['ProductName'] = products_df['ProductName'].astype(str)
products_df['Value'] = products_df['Value'].astype(float)
products_df['Weight'] = products_df['Weight'].astype(float)
products = list(products_df['ProductName'])
value = dict(zip(products_df['ProductName'], products_df['Value']))
weight = dict(zip(products_df['ProductName'], products_df['Weight']))
if set(capacity.keys()) != set(shelves):
    raise ValueError('Mismatch in shelf identifiers between capacity data and shelf list.')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between value/weight data and product list.')
m = gp.Model('Retail_Shelf_Allocation')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')