import gurobipy as gp
import pandas as pd
import numpy as np
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
df = pd.read_csv(csv_path, sep=',', dtype=str, keep_default_na=False)

def contains_zz(s):
    return 'zz' in s.strip().casefold()
zz_mask = df['SKU'].apply(contains_zz)
df_zz = df.loc[zz_mask].copy()
required_cols = ['SKU', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df_zz.columns:
        raise KeyError(f"Required column '{col}' not found in CSV.")
for col in ['Revenue', 'Demand', 'Initial Inventory']:
    try:
        if col == 'Revenue':
            df_zz[col] = df_zz[col].astype(float)
        elif col == 'Demand':
            df_zz[col] = df_zz[col].astype(int)
        elif col == 'Initial Inventory':
            df_zz[col] = df_zz[col].astype(float)
    except Exception as e:
        raise ValueError(f"Failed to convert column '{col}' to numeric: {e}")
zz_skus = df_zz['SKU'].tolist()
revenue = df_zz.set_index('SKU')['Revenue'].to_dict()
demand = df_zz.set_index('SKU')['Demand'].to_dict()
init_inventory = df_zz.set_index('SKU')['Initial Inventory'].to_dict()
upper_bound = {}
for sku in zz_skus:
    d = demand[sku]
    inv = init_inventory[sku]
    upper_bound[sku] = int(min(d, np.floor(inv)))
m = gp.Model('MaximizeZZRevenue')
x_vars = m.addVars(zz_skus, vtype=gp.GRB.INTEGER, lb=0, ub=[upper_bound[sku] for sku in zz_skus], name='')
m.setObjective(gp.quicksum((revenue[sku] * x_vars[sku] for sku in zz_skus)), gp.GRB.MAXIMIZE)
for sku in zz_skus:
    m.addConstr(x_vars[sku] <= demand[sku], name='dem_' + sku)
    m.addConstr(x_vars[sku] <= int(np.floor(init_inventory[sku])), name='inv_' + sku)
    m.addConstr(x_vars[sku] >= 0, name='nonneg_' + sku)
m.optimize()