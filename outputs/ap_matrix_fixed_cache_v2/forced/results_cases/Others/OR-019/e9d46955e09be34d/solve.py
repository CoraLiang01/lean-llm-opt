import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)

def normalize_str(s):
    return re.sub('\\s+', ' ', s.strip()).casefold()
product_name_col = 'Product Name'
product_mask = df[product_name_col].apply(lambda x: '27in' in normalize_str(x))
df_27in = df[product_mask].copy()
if df_27in.empty:
    raise ValueError("No products found with '27in' in the Product Name column.")
product_ids = df_27in[product_name_col].tolist()

def to_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to float: {e}")

def to_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{colname}' to int: {e}")
revenue_dict = dict(zip(product_ids, to_float(df_27in['Revenue'], 'Revenue')))
demand_dict = dict(zip(product_ids, to_int(df_27in['Demand'], 'Demand')))
inventory_dict = dict(zip(product_ids, to_int(df_27in['Initial Inventory'], 'Initial Inventory')))
for pid in product_ids:
    if pid not in revenue_dict or pid not in demand_dict or pid not in inventory_dict:
        raise ValueError(f"Missing parameter(s) for product '{pid}'.")
m = gp.Model('27in_Product_Fulfillment')
x_vars = m.addVars(product_ids, vtype=gp.GRB.INTEGER, lb=0, name='')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= inventory_dict[pid], name=f'inv_{pid}')
for pid in product_ids:
    m.addConstr(x_vars[pid] <= demand_dict[pid], name=f'demand_{pid}')
m.setObjective(gp.quicksum((revenue_dict[pid] * x_vars[pid] for pid in product_ids)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for '27in' Products ---")
    for pid in product_ids:
        print(f'Product: {pid}')
        print(f'  Units fulfilled: {int(round(x_vars[pid].X))}')
        print(f'  Demand: {demand_dict[pid]}')
        print(f'  Initial Inventory: {inventory_dict[pid]}')
        print(f'  Per-unit Revenue: {revenue_dict[pid]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')