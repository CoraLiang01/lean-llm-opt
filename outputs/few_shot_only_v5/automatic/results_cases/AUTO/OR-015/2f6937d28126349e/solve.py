import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM6/RestaurantSalesreport.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
pattern = re.compile('^\\s*(?P<Product_Name>[^\\n,]+)\\s*,\\s*(?P<Revenue>[^\\n,]+)\\s*,\\s*(?P<Demand>[^\\n,]+)\\s*,\\s*(?P<Initial_Inventory>[^\\n,]+)\\s*$', re.MULTILINE)
matches = list(pattern.finditer(csv_text))
if not matches:
    raise ValueError('CSV parsing failed: no matching rows found.')
header_pattern = re.compile('Product Name\\s*,\\s*Revenue\\s*,\\s*Demand\\s*,\\s*Initial Inventory', re.IGNORECASE)
if not header_pattern.search(csv_text):
    raise ValueError('CSV parsing failed: header row not found or malformed.')
products = []
revenue = {}
demand = {}
inventory = {}
for m in matches:
    prod = m.group('Product_Name').strip()
    products.append(prod)
    try:
        revenue[prod] = float(m.group('Revenue').strip())
        demand[prod] = int(m.group('Demand').strip())
        inventory[prod] = float(m.group('Initial_Inventory').strip())
    except Exception as e:
        raise ValueError(f"CSV parsing failed for product '{prod}': {e}")
if not set(products) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Mismatch in product indices among data fields.')
m = gp.Model('Aalop_Revenue_Max')
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