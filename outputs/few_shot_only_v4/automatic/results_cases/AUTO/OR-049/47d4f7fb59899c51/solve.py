import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'Shelf' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must have columns: Shelf, Capacity')
shelves = list(capacity_df['Shelf'])
capacities = {}
for idx, row in capacity_df.iterrows():
    try:
        capacities[row['Shelf']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for shelf {row['Shelf']}: {row['Capacity']}")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA15/RetailSalesAnalysis1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError(f'products.csv must have columns: {required_cols}')
products = list(products_df['ProductName'])
values = {}
weights = {}
for idx, row in products_df.iterrows():
    try:
        values[row['ProductName']] = float(row['Value'])
        weights[row['ProductName']] = float(row['Weight'])
    except Exception:
        raise ValueError(f"Invalid value/weight for product {row['ProductName']}: {row['Value']}, {row['Weight']}")
if len(shelves) == 0 or len(products) == 0:
    raise ValueError('No shelves or products found in input data.')
for s in shelves:
    if s not in capacities:
        raise ValueError(f'Missing capacity for shelf {s}')
for p in products:
    if p not in values or p not in weights:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('Retail_Shelf_Allocation')
x_vars = m.addVars(shelves, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((values[p] * x_vars[s, p] for s in shelves for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weights[p] * x_vars[s, p] for p in products)) <= capacities[s] for s in shelves), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')