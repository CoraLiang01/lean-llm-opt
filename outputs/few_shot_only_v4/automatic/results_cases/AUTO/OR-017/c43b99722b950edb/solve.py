import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM8/RetailStoreSalesTransactions(ScannerData).csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.match('^(.*)\\n', csv_text)
if not header_match:
    raise ValueError('CSV header row not found or unmatched by regex.')
header_row = header_match.group(1)
header_cols = [h.strip() for h in header_row.split(',')]
required_cols = ['Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in header_cols:
        raise ValueError(f"Required column '{col}' not found in CSV header.")
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
if 'ZZ' not in df.values:
    raise ValueError("No product classified under 'ZZ' found in the dataset.")
classification_col = None
for col in df.columns:
    if df[col].str.strip().eq('ZZ').any():
        classification_col = col
        break
if classification_col is None:
    raise ValueError("No column found with 'ZZ' classification.")
zz_mask = df[classification_col].str.strip() == 'ZZ'
df_zz = df.loc[zz_mask].copy()
if df_zz.empty:
    raise ValueError("No rows found for products classified under 'ZZ'.")
product_keys = df_zz.index.tolist()

def parse_numeric_column(series, colname):
    vals = {}
    for idx, val in series.items():
        val_str = str(val).strip()
        if not re.match('^-?\\d+(\\.\\d+)?$', val_str):
            raise ValueError(f"Non-numeric or unmatched value '{val}' in column '{colname}' at row {idx}")
        vals[idx] = float(val_str)
    return vals
revenue = parse_numeric_column(df_zz['Revenue'], 'Revenue')
demand = parse_numeric_column(df_zz['Demand'], 'Demand')
initial_inventory = parse_numeric_column(df_zz['Initial Inventory'], 'Initial Inventory')
for k in product_keys:
    if k not in revenue or k not in demand or k not in initial_inventory:
        raise ValueError(f'Missing parameter for product index {k}')
m = gp.Model('Retail_ZZ_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_keys, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_keys)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in product_keys), name='')
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in product_keys), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')