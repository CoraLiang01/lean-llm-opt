import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM11/SalesDatainBusinesses.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.match('\\s*Product Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*\\n', csv_text)
if not header_match:
    raise ValueError('CSV header does not match expected format.')
row_pattern = re.compile('^\\s*([^\\s]+)\\s+([0-9.]+)\\s+([0-9]+)\\s+([0-9]+)\\s*$', re.MULTILINE)
rows = row_pattern.findall(csv_text)
if not rows or len(rows) == 0:
    raise ValueError('No data rows matched in CSV.')
products = []
revenue = {}
demand = {}
inventory = {}
for row in rows:
    prod, rev, dem, inv = row
    products.append(prod)
    try:
        revenue[prod] = float(rev)
        demand[prod] = int(dem)
        inventory[prod] = int(inv)
    except Exception as e:
        raise ValueError(f'Error parsing row for product {prod}: {e}')
if not set(revenue) == set(demand) == set(inventory) == set(products):
    raise ValueError('Mismatch in product keys among revenue, demand, and inventory.')
m = gp.Model('Supermarket_Revenue_Max')
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')