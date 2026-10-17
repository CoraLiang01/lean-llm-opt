import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM19/WomenClothingEcommerceSalesData.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
header_match = re.match('^\\s*Product Name\\s+Revenue\\s+Demand\\s+Initial Inventory\\s*$', csv_text.splitlines()[0])
if not header_match:
    raise ValueError('CSV header does not match expected columns.')
row_pattern = re.compile('^\\s*(?P<product>[^\\s]+)\\s+(?P<revenue>\\d+)\\s+(?P<demand>\\d+)\\s+(?P<inventory>\\d+)\\s*$')
lines = csv_text.splitlines()[1:]
products = []
revenue = {}
demand = {}
inventory = {}
for line in lines:
    m = row_pattern.match(line)
    if not m:
        raise ValueError(f'CSV row does not match expected format: {line!r}')
    prod = m.group('product')
    products.append(prod)
    revenue[prod] = int(m.group('revenue'))
    demand[prod] = int(m.group('demand'))
    inventory[prod] = int(m.group('inventory'))
if not set(products) == set(revenue) == set(demand) == set(inventory):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('WomenClothingEcommerceSales')
x_vars = m.addVars(products, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')