import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def to_float_or_raise(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e

def to_int_or_raise(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-integer values.") from e
revenue = to_float_or_raise(df['Revenue'], 'Revenue')
demand = to_int_or_raise(df['Demand'], 'Demand')
init_inventory = to_int_or_raise(df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(product_ids, revenue))
demand_dict = dict(zip(product_ids, demand))
inventory_dict = dict(zip(product_ids, init_inventory))
m = gp.Model('MobileDeviceOrderFulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand_dict[pid], name=f'demand_{pid}')
    m.addConstr(x_vars[pid] <= inventory_dict[pid], name=f'inventory_{pid}')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for pid in product_ids:
        fulfilled = int(round(x_vars[pid].X))
        print(f'Product: {pid} | Fulfilled: {fulfilled} | Demand: {demand_dict[pid]} | Inventory: {inventory_dict[pid]} | Revenue/unit: {revenue_dict[pid]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')