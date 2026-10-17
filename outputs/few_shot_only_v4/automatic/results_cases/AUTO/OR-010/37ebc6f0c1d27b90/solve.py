import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(.*)$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('CSV header row not found or unmatched.')
header_row = header_match.group(1)
header_cols = [h.strip() for h in header_row.split(',')]
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in header_cols:
        raise ValueError(f"Required column '{col}' not found in CSV header.")
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV data.")
products = df['Product Name'].tolist()
if len(products) != len(df):
    raise ValueError('Mismatch in number of products and rows.')
try:
    revenue = {row['Product Name']: float(row['Revenue']) for _, row in df.iterrows()}
    demand = {row['Product Name']: int(float(row['Demand'])) for _, row in df.iterrows()}
    inventory = {row['Product Name']: int(float(row['Initial Inventory'])) for _, row in df.iterrows()}
except Exception as e:
    raise ValueError(f'Error parsing numeric fields: {e}')
for i in products:
    if i not in revenue or i not in demand or i not in inventory:
        raise ValueError(f"Missing data for product '{i}'.")
m = gp.Model('MobileSalesNRM')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')