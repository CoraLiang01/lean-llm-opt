import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM22/DairyGoodsSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(.*?)(?:\\r?\\n)', csv_text)
if not header_match:
    raise ValueError('CSV header row not found.')
header_row = header_match.group(1)
header_fields = [h.strip() for h in header_row.split(',')]
required_fields = ['Full_Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
for field in required_fields:
    if field not in header_fields:
        raise ValueError(f"Required column '{field}' not found in CSV header.")
pattern = re.compile('^(?P<Full_Product_Name>[^,]*),(?P<Revenue>[^,]*),(?P<Demand>[^,]*),(?P<Initial Inventory>[^,]*)(?:,.*)?$')
lines = csv_text.strip().split('\n')
data_rows = []
for line in lines[1:]:
    m = pattern.match(line)
    if not m:
        raise ValueError(f'CSV row does not match expected format: {line}')
    data_rows.append(m.groupdict())
if not data_rows:
    raise ValueError('No data rows found in CSV.')
items = []
revenue = {}
demand = {}
inventory = {}
for row in data_rows:
    prod = row['Full_Product_Name'].strip()
    if prod in items:
        raise ValueError(f'Duplicate product identifier: {prod}')
    try:
        rev = float(row['Revenue'].strip())
        dem = float(row['Demand'].strip())
        inv = float(row['Initial Inventory'].strip())
    except Exception as e:
        raise ValueError(f"Non-numeric value in row for product '{prod}': {e}")
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
for prod in items:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product: {prod}')
m = gp.Model('DairyGoodsOrderFulfillment')
x_vars = m.addVars(items, lb=0.0, vtype=GRB.CONTINUOUS, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')