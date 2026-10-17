import gurobipy as gp
from gurobipy import GRB
import pandas as pd
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA20/BigMartSalesData2/products.csv', dtype=str, keep_default_na=False)
shelves = [row['ShelfID'] for _, row in capacity_df.iterrows()]
products = [row['ProductName'] for _, row in products_df.iterrows()]
capacity = {}
for _, row in capacity_df.iterrows():
    try:
        capacity[row['ShelfID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid Capacity for ShelfID {row['ShelfID']}")
value = {}
weight = {}
for _, row in products_df.iterrows():
    try:
        value[row['ProductName']] = float(row['Value'])
        weight[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid Value or Weight for ProductName {row['ProductName']}")
if len(shelves) != 10:
    raise ValueError('Expected 10 shelves from capacity.csv')
if len(products) != 20:
    raise ValueError('Expected 20 products from products.csv')
if set(capacity.keys()) != set(shelves):
    raise ValueError('Mismatch in shelf identifiers between capacity.csv and shelves list')
if set(value.keys()) != set(products) or set(weight.keys()) != set(products):
    raise ValueError('Mismatch in product identifiers between products.csv and value/weight dictionaries')
m = gp.Model('BigMartSales')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')