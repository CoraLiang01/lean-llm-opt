import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
product_ids = df['Product Name'].tolist()

def to_float_series(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to float: {e}")

def to_int_series(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' could not be converted to int: {e}")
revenue = to_float_series(df['Revenue'], 'Revenue')
demand = to_int_series(df['Demand'], 'Demand')
initial_inventory = to_int_series(df['Initial Inventory'], 'Initial Inventory')
revenue_dict = dict(zip(product_ids, revenue))
demand_dict = dict(zip(product_ids, demand))
inventory_dict = dict(zip(product_ids, initial_inventory))
if not set(revenue_dict.keys()) == set(demand_dict.keys()) == set(inventory_dict.keys()) == set(product_ids):
    raise ValueError('Mismatch in product IDs across parameter dictionaries.')
m = gp.Model('SupermarketRevenueMaximization')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
m.addConstrs((x_vars[i] <= demand_dict[i] for i in product_ids), name='')
m.addConstrs((x_vars[i] <= inventory_dict[i] for i in product_ids), name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total value/cost: {m.objVal:.2f}')
    print('--- Fulfillment Plan ---')
    for i in product_ids:
        print(f'Product {i}: Fulfilled {int(round(x_vars[i].X))} units (Demand: {demand_dict[i]}, Inventory: {inventory_dict[i]}, Revenue/unit: {revenue_dict[i]})')
else:
    print(f'No optimal solution found. Status: {m.status}')