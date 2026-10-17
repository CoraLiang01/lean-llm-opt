import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM3/OnlineSalesDataset.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
product_names = []
revenue = {}
demand = {}
inventory = {}
row_regex = re.compile('.+')
for idx, row in df.iterrows():
    pname = row['Product Name']
    if not row_regex.match(pname):
        raise ValueError(f"Unmatched product name in row {idx}: {row['Product Name']}")
    try:
        rev = float(row['Revenue'])
    except Exception:
        raise ValueError(f"Unmatched or invalid Revenue in row {idx}: {row['Revenue']}")
    try:
        dem = int(float(row['Demand']))
    except Exception:
        raise ValueError(f"Unmatched or invalid Demand in row {idx}: {row['Demand']}")
    try:
        inv = int(float(row['Initial Inventory']))
    except Exception:
        raise ValueError(f"Unmatched or invalid Initial Inventory in row {idx}: {row['Initial Inventory']}")
    product_names.append(pname)
    revenue[pname] = rev
    demand[pname] = dem
    inventory[pname] = inv
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(product_names):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('OnlineRetailer_InventoryRevenue')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in product_names), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in product_names), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')