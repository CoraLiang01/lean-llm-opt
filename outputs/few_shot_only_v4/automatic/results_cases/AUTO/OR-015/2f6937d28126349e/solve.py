import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
if not os.path.exists(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(.*\\n)', csv_text)
if not header_match:
    raise ValueError('CSV header row not found or unmatched.')
header_line = header_match.group(1).strip()
header_cols = [h.strip() for h in header_line.split(',')]
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in header_cols:
        raise ValueError(f"Required column '{col}' not found in CSV header.")
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV data.")
if 'Product Name' not in df.columns:
    raise ValueError("Column 'Product Name' not found in CSV.")
aalop_mask = df['Product Name'].notnull()
aalop_df = df.loc[aalop_mask].copy()
if aalop_df.empty:
    raise ValueError("No products classified under 'Aalop' found in CSV.")
items = list(aalop_df['Product Name'])
if len(set(items)) != len(items):
    raise ValueError("Duplicate product names found in 'Product Name' column.")

def parse_numeric_column(series, colname):
    vals = {}
    for idx, (k, v) in enumerate(zip(aalop_df['Product Name'], series)):
        v_stripped = str(v).strip()
        if not re.fullmatch('-?\\d+(\\.\\d+)?', v_stripped):
            raise ValueError(f"Non-numeric or unmatched value '{v}' in column '{colname}' for product '{k}' (row {idx + 2}).")
        try:
            val = float(v_stripped)
        except Exception:
            raise ValueError(f"Could not convert value '{v}' in column '{colname}' for product '{k}' to float.")
        vals[k] = val
    return vals
revenue = parse_numeric_column(aalop_df['Revenue'], 'Revenue')
demand = parse_numeric_column(aalop_df['Demand'], 'Demand')
inventory = parse_numeric_column(aalop_df['Initial Inventory'], 'Initial Inventory')
for k in items:
    if k not in revenue or k not in demand or k not in inventory:
        raise ValueError(f"Missing data for product '{k}' in one or more required columns.")
for k in items:
    if demand[k] < 0 or inventory[k] < 0:
        raise ValueError(f"Negative demand or inventory for product '{k}'.")
m = gp.Model('Aalop_Restaurant_RevOpt')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[k] * x_vars[k] for k in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[k] <= demand[k] for k in items), name='')
m.addConstrs((x_vars[k] <= inventory[k] for k in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')