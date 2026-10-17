import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import numpy as np
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM18/SupermartGrocerySales-RetailAnalyticsDataset.csv'

def parse_csv_with_regex(path):
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    header = lines[0].rstrip('\n').split(',')
    colmap = {col: idx for idx, col in enumerate(header)}
    required_cols = ['Product', 'Category', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in colmap:
            raise ValueError(f"Required column '{col}' not found in CSV header.")
    data = []
    regex = re.compile('^(?P<Product>.*?),(?P<Category>.*?),(?P<Revenue>.*?),(?P<Demand>.*?),(?P<Initial Inventory>.*?)(,.*)?$')
    for i, line in enumerate(lines[1:], 2):
        line = line.rstrip('\n')
        m = regex.match(line)
        if not m:
            raise ValueError(f'Line {i} does not match expected CSV structure: {line}')
        row = {k: m.group(k) for k in required_cols}
        data.append(row)
    return data
raw_data = parse_csv_with_regex(csv_path)
organ_products = []
revenue = {}
demand = {}
inventory = {}
for row in raw_data:
    if row['Category'].strip() == 'Organ':
        prod = row['Product'].strip()
        try:
            rev = float(row['Revenue'])
            dem = float(row['Demand'])
            inv = float(row['Initial Inventory'])
        except Exception as e:
            raise ValueError(f"Non-numeric value in row for product '{prod}': {e}")
        if np.isnan(rev) or np.isnan(dem) or np.isnan(inv):
            raise ValueError(f"NaN value in row for product '{prod}'")
        organ_products.append(prod)
        revenue[prod] = rev
        demand[prod] = int(dem)
        inventory[prod] = int(inv)
if not organ_products:
    raise ValueError("No products classified under 'Organ' found in the CSV.")
for prod in organ_products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f"Missing data for product '{prod}'.")
m = gp.Model('Supermart_Organ_Revenue')
x_vars = m.addVars(organ_products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in organ_products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in organ_products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in organ_products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')