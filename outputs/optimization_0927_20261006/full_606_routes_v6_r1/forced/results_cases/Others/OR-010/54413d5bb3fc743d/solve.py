import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def to_float_or_error(val, col, pid):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Invalid float in column '{col}' for product '{pid}': '{val}'")

def to_int_or_error(val, col, pid):
    try:
        return int(val)
    except Exception:
        raise ValueError(f"Invalid int in column '{col}' for product '{pid}': '{val}'")
revenue = {}
demand = {}
initial_inventory = {}
for (idx, row) in df.iterrows():
    pid = row['Product Name']
    revenue[pid] = to_float_or_error(row['Revenue'], 'Revenue', pid)
    demand[pid] = to_int_or_error(row['Demand'], 'Demand', pid)
    initial_inventory[pid] = to_int_or_error(row['Initial Inventory'], 'Initial Inventory', pid)
m = gp.Model('MobileDeviceOrderFulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand[pid], name=f'demand_limit_{pid}')
    m.addConstr(x_vars[pid] <= initial_inventory[pid], name=f'inventory_limit_{pid}')
m.setObjective(gp.quicksum((revenue[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pid in product_ids:
        fulfilled = int(round(x_vars[pid].X))
        print(f'Product: {pid}')
        print(f'  Fulfilled: {fulfilled} units')
        print(f'  Demand: {demand[pid]}')
        print(f'  Initial Inventory: {initial_inventory[pid]}')
        print(f'  Per-unit Revenue: {revenue[pid]:.2f}')
        print(f'  Revenue from this product: {revenue[pid] * fulfilled:.2f}')
        print()
else:
    print(f'No optimal solution found. Status: {m.status}')