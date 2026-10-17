import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.search('^\\s*Product Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text, re.MULTILINE)
if not header_match:
    raise ValueError('Header row not found or does not match required columns.')
row_pattern = re.compile('^\\s*(?P<product>S700_\\S+)\\s+(?P<revenue>[0-9.]+)\\s+(?P<demand>[0-9]+)\\s+(?P<inventory>[0-9]+)\\s*$', re.MULTILINE)
matches = list(row_pattern.finditer(csv_text))
if not matches:
    raise ValueError('No matching S700_ products found in the CSV.')
products = []
revenue = {}
demand = {}
inventory = {}
for m in matches:
    prod = m.group('product')
    products.append(prod)
    try:
        revenue[prod] = float(m.group('revenue'))
        demand[prod] = int(m.group('demand'))
        inventory[prod] = int(m.group('inventory'))
    except Exception as e:
        raise ValueError(f'Failed to parse numeric fields for product {prod}: {e}')
for prod in products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product {prod}')
m = gp.Model('Retail_S700_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')