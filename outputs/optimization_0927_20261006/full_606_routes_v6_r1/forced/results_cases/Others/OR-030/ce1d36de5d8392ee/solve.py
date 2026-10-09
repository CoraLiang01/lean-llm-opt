import gurobipy as gp
import pandas as pd
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
product_name_col = 'Product Name'
fdk57_mask = df[product_name_col].str.strip().str.casefold() == 'fdk57'
df_fdk57 = df[fdk57_mask].copy()
if df_fdk57.empty:
    raise ValueError("No car models with 'Product Name' exactly 'FDK57' found in the data.")
fdk57_indices = df_fdk57.index.tolist()

def to_float(val, row_idx, col):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Invalid float in column '{col}' at row {row_idx}: {val}")

def to_int(val, row_idx, col):
    try:
        return int(val)
    except Exception:
        raise ValueError(f"Invalid int in column '{col}' at row {row_idx}: {val}")
revenue_col = 'Revenue'
demand_col = 'Demand'
inventory_col = 'Initial Inventory'
revenue = {idx: to_float(df_fdk57.at[idx, revenue_col], idx, revenue_col) for idx in fdk57_indices}
demand = {idx: to_int(df_fdk57.at[idx, demand_col], idx, demand_col) for idx in fdk57_indices}
initial_inventory = {idx: to_int(df_fdk57.at[idx, inventory_col], idx, inventory_col) for idx in fdk57_indices}
m = gp.Model('FDK57_Fulfillment')
x_vars = m.addVars(fdk57_indices, vtype=gp.GRB.INTEGER, lb=0, name='')
for idx in fdk57_indices:
    m.addConstr(x_vars[idx] <= demand[idx], name=f'demand_{idx}')
    m.addConstr(x_vars[idx] <= initial_inventory[idx], name=f'inventory_{idx}')
m.setObjective(gp.quicksum((revenue[idx] * x_vars[idx] for idx in fdk57_indices)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print('--- Fulfillment Plan for FDK57 Models ---')
    for idx in fdk57_indices:
        fulfilled = int(round(x_vars[idx].X))
        print(f'Row {idx}: FDK57 | Revenue/unit: {revenue[idx]:.2f} | Demand: {demand[idx]} | Inventory: {initial_inventory[idx]} | Fulfilled: {fulfilled}')
else:
    print(f'No optimal solution found. Status: {m.status}')