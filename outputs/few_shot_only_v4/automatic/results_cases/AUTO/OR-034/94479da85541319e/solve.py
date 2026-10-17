import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM25/Frenchbakerydailysales.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(.*Product Name.*)$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError("CSV header with 'Product Name' not found.")
header_line = header_match.group(1)
header_cols = [h.strip() for h in header_line.split(',')]
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in header_cols:
        raise ValueError(f"Required column '{col}' not found in CSV header.")
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV data.")
product_names = df['Product Name'].tolist()
if len(product_names) != len(set(product_names)):
    raise ValueError("Duplicate product names found in 'Product Name' column.")
try:
    revenue_dict = {row['Product Name']: float(row['Revenue']) for _, row in df.iterrows()}
    demand_dict = {row['Product Name']: int(row['Demand']) for _, row in df.iterrows()}
    inventory_dict = {row['Product Name']: int(row['Initial Inventory']) for _, row in df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing numeric fields: {e}')
for pname in product_names:
    if pname not in revenue_dict or pname not in demand_dict or pname not in inventory_dict:
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('FrenchBakeryRevenueMax')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue_dict[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand_dict[i] for i in product_names), name='')
m.addConstrs((x_vars[i] <= inventory_dict[i] for i in product_names), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')