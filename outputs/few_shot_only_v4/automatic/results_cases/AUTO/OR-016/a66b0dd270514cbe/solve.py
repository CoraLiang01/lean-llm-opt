import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM7/RetailSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^(.*)$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('CSV header not found.')
header = header_match.group(1).strip().split(',')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in header:
        raise ValueError(f"Required column '{col}' not found in CSV header.")
rows = []
for m in re.finditer('^(.*)$', csv_text, re.MULTILINE):
    line = m.group(1)
    if line.strip() == '' or line.strip() == ','.join(header):
        continue
    fields = [x.strip() for x in re.split(',(?=(?:[^"]*"[^"]*")*[^"]*$)', line)]
    if len(fields) != len(header):
        continue
    rows.append(dict(zip(header, fields)))
if not rows:
    raise ValueError('No data rows found in CSV.')
products = []
revenue = {}
demand = {}
inventory = {}
for row in rows:
    pname = row['Product Name']
    try:
        rev = float(row['Revenue'])
        dem = float(row['Demand'])
        inv = float(row['Initial Inventory'])
    except Exception:
        raise ValueError(f"Non-numeric value in row for product '{pname}'.")
    products.append(pname)
    revenue[pname] = rev
    demand[pname] = dem
    inventory[pname] = inv
if len(set(products)) != len(products):
    raise ValueError('Duplicate product names found.')
for pname in products:
    if not (pname in revenue and pname in demand and (pname in inventory)):
        raise ValueError(f"Missing data for product '{pname}'.")
m = gp.Model('Retail_Allocation')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, ub=[min(demand[p], inventory[p]) for p in products], vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[p] * x_vars[p] for p in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[p] <= demand[p] for p in products), name='')
m.addConstrs((x_vars[p] <= inventory[p] for p in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')