import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM14/SalesStoreoverview.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
if not set(['Product_Reference', 'Revenue', 'Demand', 'Initial Inventory']).issubset(df.columns):
    raise ValueError('CSV missing required columns.')
ele_s_mask = df['Product_Reference'].str.match('\\s*ELE-S')
ele_s_df = df[ele_s_mask].copy()
if ele_s_df.empty:
    raise ValueError("No products with Product_Reference starting with 'ELE-S' found.")
product_indices = []
revenue = {}
demand = {}
inventory = {}
for idx, row in ele_s_df.iterrows():
    prod = row['Product_Reference'].strip()
    try:
        r = float(row['Revenue'])
        d = int(float(row['Demand']))
        s = int(float(row['Initial Inventory']))
    except Exception as e:
        raise ValueError(f'Failed to parse numeric fields for product {prod}: {e}')
    product_indices.append(prod)
    revenue[prod] = r
    demand[prod] = d
    inventory[prod] = s
if not set(product_indices) == set(revenue) == set(demand) == set(inventory):
    raise ValueError('Mismatch in product indices and parameter dictionaries.')
m = gp.Model('ELE_S_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_indices, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_indices)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in product_indices), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in product_indices), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')