import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/capacity.csv', dtype=str, keep_default_na=False)
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA19/BigMartSalesData1/products.csv', dtype=str, keep_default_na=False)
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
if set(shelves) != set((str(i) for i in range(1, 11))):
    raise ValueError('ShelfID set does not match expected 1-10')
if set(products) != set((str(j) for j in range(1, 21))):
    raise ValueError('ProductName set does not match expected 1-20')
if not all((k in value and k in weight for k in products)):
    raise ValueError('Missing value or weight for some products')
if not all((k in capacity for k in shelves)):
    raise ValueError('Missing capacity for some shelves')
m = gp.Model('BigMart_Shelf_Allocation')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[j] * x_vars[i, j] for i in shelves for j in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[j] * x_vars[i, j] for j in products)) <= capacity[i] for i in shelves), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')