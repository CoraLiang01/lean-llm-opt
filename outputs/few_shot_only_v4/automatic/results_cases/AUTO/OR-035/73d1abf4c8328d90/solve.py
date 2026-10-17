import pandas as pd
import gurobipy as gp
from gurobipy import GRB
capacity_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/capacity.csv'
products_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/RA_testing/RA1/BakerySales1/products.csv'
capacity_df = pd.read_csv(capacity_path, dtype=str, keep_default_na=False)
products_df = pd.read_csv(products_path, dtype=str, keep_default_na=False)
if 'BreadType' in products_df.columns:
    bread_types = list(products_df['BreadType'])
else:
    raise ValueError("products.csv must contain a 'BreadType' column.")
if 'Value' in products_df.columns:
    profit = {}
    for idx, row in products_df.iterrows():
        try:
            profit[row['BreadType']] = float(row['Value'])
        except Exception:
            raise ValueError(f"Invalid Value for bread type {row['BreadType']}")
else:
    raise ValueError("products.csv must contain a 'Value' column.")
if 'Weight' in products_df.columns:
    weight = {}
    for idx, row in products_df.iterrows():
        try:
            weight[row['BreadType']] = float(row['Weight'])
        except Exception:
            raise ValueError(f"Invalid Weight for bread type {row['BreadType']}")
else:
    raise ValueError("products.csv must contain a 'Weight' column.")
capacity_col = None
for col in capacity_df.columns:
    if col.lower() in ['capacity', 'totalcapacity', 'storagecapacity']:
        capacity_col = col
        break
if capacity_col is None:
    raise ValueError("capacity.csv must contain a 'Capacity' or 'TotalCapacity' column.")
try:
    total_capacity = float(capacity_df[capacity_col].iloc[0])
except Exception:
    raise ValueError('Invalid value in capacity.csv for total capacity.')
for b in bread_types:
    if b not in profit or b not in weight:
        raise ValueError(f'Missing profit or weight for bread type {b}')
m = gp.Model('Bakery_Bread_Order')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(bread_types, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((profit[b] * x_vars[b] for b in bread_types)), GRB.MAXIMIZE)
m.addConstr(gp.quicksum((weight[b] * x_vars[b] for b in bread_types)) <= total_capacity, name='storage')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')