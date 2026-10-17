import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/capacity.csv', dtype=str, keep_default_na=False)
if 'ShelfID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must have columns ShelfID and Capacity')
shelves = list(capacity_df['ShelfID'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['ShelfID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for shelf {row['ShelfID']}")
products_df = pd.read_csv('/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/Mixture_testing/Mixture1/RetailSalesAnalysis/products.csv', dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must have columns ProductName, Value, Weight')
products = list(products_df['ProductName'])
value = {}
weight = {}
for idx, row in products_df.iterrows():
    try:
        value[row['ProductName']] = float(row['Value'])
        weight[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid value/weight for product {row['ProductName']}")
if len(products) < 1:
    raise ValueError('No products found in products.csv')
if len(shelves) < 1:
    raise ValueError('No shelves found in capacity.csv')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
for s in shelves:
    if s not in capacity:
        raise ValueError(f'Missing capacity for shelf {s}')
m = gp.Model('Retail_Display_Allocation')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
for s in shelves:
    m.addConstr(gp.quicksum((weight[p] * x_vars[s, p] for p in products)) <= capacity[s], name=f'cap_{s}')
first_product = products[0]
m.addConstr(gp.quicksum((x_vars[s, first_product] for s in shelves)) >= 5, name='min_smartphone')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for s in shelves:
        for p in products:
            var = x_vars[s, p]
            print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')