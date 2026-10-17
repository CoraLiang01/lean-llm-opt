import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
pattern = re.compile('^\\s*(?P<Product_Name>[^\\t\\n\\r\\f\\v]+?)\\s+(?P<Revenue>-?\\d+(?:\\.\\d+)?)\\s+(?P<Demand>-?\\d+)\\s+(?P<Initial_Inventory>-?\\d+)\\s*$', re.MULTILINE)
matches = list(pattern.finditer(csv_text))
if not matches:
    raise ValueError('No rows matched the expected CSV pattern. Check file format.')
products = []
revenue = {}
demand = {}
inventory = {}
for m in matches:
    prod = m.group('Product_Name').strip()
    if prod in products:
        raise ValueError(f'Duplicate product name found: {prod}')
    products.append(prod)
    try:
        revenue[prod] = float(m.group('Revenue'))
        demand[prod] = int(m.group('Demand'))
        inventory[prod] = int(m.group('Initial_Inventory'))
    except Exception as e:
        raise ValueError(f'Error parsing numeric fields for product {prod}: {e}')
for prod in products:
    if prod not in revenue or prod not in demand or prod not in inventory:
        raise ValueError(f'Missing data for product {prod}')
m = gp.Model('Pizza_Fulfillment')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[prod] * x_vars[prod] for prod in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[prod] <= inventory[prod] for prod in products), name='')
m.addConstrs((x_vars[prod] <= demand[prod] for prod in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in x_vars.values():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')