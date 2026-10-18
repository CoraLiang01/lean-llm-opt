import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def sku_has_zz(sku):
    return 'zz' in sku.strip().casefold()
zz_mask = df['SKU'].apply(sku_has_zz)
zz_df = df.loc[zz_mask].copy()
required_columns = ['SKU', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in zz_df.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
zz_df['SKU'] = zz_df['SKU'].astype(str)
zz_df.set_index('SKU', inplace=True, drop=False)
zz_products = list(zz_df.index)

def safe_float(series, colname):
    try:
        return series.astype(float)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-numeric values.") from e

def safe_int(series, colname):
    try:
        return series.astype(int)
    except Exception as e:
        raise ValueError(f"Column '{colname}' contains non-integer values.") from e
revenue = safe_float(zz_df['Revenue'], 'Revenue').to_dict()
demand = safe_int(zz_df['Demand'], 'Demand').to_dict()
initial_inventory = safe_float(zz_df['Initial Inventory'], 'Initial Inventory').to_dict()
for sku in zz_products:
    if sku not in revenue or sku not in demand or sku not in initial_inventory:
        raise KeyError(f"Missing parameter for SKU '{sku}'.")
m = gp.Model('RetailStore_ZZ_Revenue_Max')
x_vars = m.addVars(zz_products, lb=0.0, ub=[min(initial_inventory[sku], demand[sku]) for sku in zz_products], vtype=gp.GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[sku] * x_vars[sku] for sku in zz_products)), gp.GRB.MAXIMIZE)
m.optimize()
if m.status == gp.GRB.OPTIMAL:
    print(f'Optimal total revenue: {m.objVal:.2f}')
    print("--- Fulfillment Plan for 'ZZ' Products ---")
    for sku in zz_products:
        fulfilled = x_vars[sku].X
        print(f'SKU: {sku} | Fulfilled: {fulfilled:.2f} | Demand: {demand[sku]} | Inventory: {initial_inventory[sku]} | Revenue/unit: {revenue[sku]:.2f}')
else:
    print(f'No optimal solution found. Status: {m.status}')