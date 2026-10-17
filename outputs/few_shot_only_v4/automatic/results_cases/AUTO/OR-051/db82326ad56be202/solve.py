import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/capacity.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
if 'CabinetID' not in capacity_df.columns or 'Capacity' not in capacity_df.columns:
    raise ValueError('capacity.csv must have columns CabinetID and Capacity')
cabinets = list(capacity_df['CabinetID'])
capacity = {}
for idx, row in capacity_df.iterrows():
    try:
        capacity[row['CabinetID']] = float(row['Capacity'])
    except Exception:
        raise ValueError(f"Invalid capacity value for cabinet {row['CabinetID']}")
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA17/CoffeeChainSalesAnalysis1/products.csv'
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
required_cols = {'ProductID', 'ProductName', 'Value', 'Weight'}
if not required_cols.issubset(products_df.columns):
    raise ValueError('products.csv must have columns ProductID, ProductName, Value, Weight')
products = list(products_df['ProductID'])
product_names = {}
value = {}
weight = {}
for idx, row in products_df.iterrows():
    pid = row['ProductID']
    product_names[pid] = row['ProductName']
    try:
        value[pid] = float(row['Value'])
        weight[pid] = float(row['Weight'])
    except Exception:
        raise ValueError(f'Invalid value or weight for product {pid}')
if len(cabinets) == 0 or len(products) == 0:
    raise ValueError('No cabinets or products found in input files')
for c in cabinets:
    if c not in capacity:
        raise ValueError(f'Missing capacity for cabinet {c}')
for p in products:
    if p not in value or p not in weight:
        raise ValueError(f'Missing value or weight for product {p}')
m = gp.Model('CoffeeCabinetAllocation')
x_vars = m.addVars(cabinets, products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((value[p] * x_vars[c, p] for c in cabinets for p in products)), GRB.MAXIMIZE)
m.addConstrs((gp.quicksum((weight[p] * x_vars[c, p] for p in products)) <= capacity[c] for c in cabinets), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')