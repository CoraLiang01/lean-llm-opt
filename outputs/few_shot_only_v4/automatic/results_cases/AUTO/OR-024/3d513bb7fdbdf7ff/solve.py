import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
product_regex = re.compile('^S700_')
mask = df['Product Name'].apply(lambda x: bool(product_regex.match(x)))
if not mask.any():
    raise ValueError("No products found with prefix 'S700_' in 'Product Name'.")
filtered = df.loc[mask, required_columns].copy()
items = []
revenue = {}
demand = {}
inventory = {}
for idx, row in filtered.iterrows():
    prod = row['Product Name']
    if prod in items:
        raise ValueError(f'Duplicate product identifier: {prod}')
    try:
        A_i = float(row['Revenue'])
        d_i = float(row['Demand'])
        I_i = float(row['Initial Inventory'])
    except Exception as e:
        raise ValueError(f"Non-numeric value in row for product '{prod}': {e}")
    if not (re.match('^-?\\d+(\\.\\d+)?$', row['Revenue'].strip()) and re.match('^-?\\d+(\\.\\d+)?$', row['Demand'].strip()) and re.match('^-?\\d+(\\.\\d+)?$', row['Initial Inventory'].strip())):
        raise ValueError(f"Unmatched numeric clause in row for product '{prod}'.")
    items.append(prod)
    revenue[prod] = A_i
    demand[prod] = d_i
    inventory[prod] = I_i
for prod in items:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f"Missing data for product '{prod}'.")
m = gp.Model('S700_Product_Revenue_Maximization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')