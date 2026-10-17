import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^\\s*SKU\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise RuntimeError('CSV header not found or does not match expected columns.')
row_pattern = re.compile('^\\s*(?P<SKU>\\S+)\\s+(?P<Revenue>[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s+(?P<Demand>[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s+(?P<InitialInventory>[-+]?\\d*\\.?\\d+(?:[eE][-+]?\\d+)?)\\s*$', re.MULTILINE)
rows = [m.groupdict() for m in row_pattern.finditer(csv_text)]
if not rows:
    raise RuntimeError('No data rows matched the expected pattern in the CSV.')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
for col in ['SKU', 'Revenue', 'Demand', 'Initial Inventory']:
    if col not in df.columns:
        raise RuntimeError(f'Missing required column: {col}')
category_col = None
for possible in ['Category', 'Product Category', 'ProductCategory', 'ProdCategory']:
    if possible in df.columns:
        category_col = possible
        break
if category_col is None:
    raise RuntimeError("No product category column found in the CSV; cannot filter for 'ZZ' products as required.")
zz_mask = df[category_col] == 'ZZ'
if not zz_mask.any():
    raise RuntimeError("No products classified under 'ZZ' found in the CSV.")
zz_df = df.loc[zz_mask].copy()
if zz_df.empty:
    raise RuntimeError("No 'ZZ' category products found after filtering.")

def parse_float_series(series, name):
    try:
        return series.astype(float)
    except Exception:
        raise RuntimeError(f"Failed to parse column '{name}' as float.")

def parse_int_series(series, name):
    try:
        return series.astype(int)
    except Exception:
        raise RuntimeError(f"Failed to parse column '{name}' as int.")
sku_list = zz_df['SKU'].tolist()
revenue_dict = dict(zip(sku_list, parse_float_series(zz_df['Revenue'], 'Revenue')))
demand_dict = dict(zip(sku_list, parse_int_series(zz_df['Demand'], 'Demand')))
inventory_dict = dict(zip(sku_list, parse_float_series(zz_df['Initial Inventory'], 'Initial Inventory')))
for sku in sku_list:
    if sku not in revenue_dict or sku not in demand_dict or sku not in inventory_dict:
        raise RuntimeError(f'Missing data for SKU {sku}.')
m = gp.Model('RetailStore_ZZ_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(sku_list, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue_dict[sku] * x_vars[sku] for sku in sku_list)), GRB.MAXIMIZE)
m.addConstrs((x_vars[sku] <= demand_dict[sku] for sku in sku_list), name='')
m.addConstrs((x_vars[sku] <= inventory_dict[sku] for sku in sku_list), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')