import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM1/MobileSalesDataset.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
pattern = '^\\s*(?P<Product_Name>.+?)\\s+(?P<Revenue>[0-9.]+)\\s+(?P<Demand>[0-9]+)\\s+(?P<Initial_Inventory>[0-9]+)\\s*$'
lines = csv_text.strip().split('\n')
header = lines[0]
data_lines = lines[1:]
matches = []
for line in data_lines:
    m = re.match(pattern, line)
    if not m:
        raise ValueError(f'CSV line did not match expected pattern: {line}')
    matches.append(m.groupdict())
product_names = []
revenue = {}
demand = {}
initial_inventory = {}
for row in matches:
    pname = row['Product_Name'].strip()
    product_names.append(pname)
    try:
        revenue[pname] = float(row['Revenue'])
        demand[pname] = int(row['Demand'])
        initial_inventory[pname] = int(row['Initial_Inventory'])
    except Exception as e:
        raise ValueError(f'Error parsing row for product {pname}: {e}')
if not set(product_names) == set(revenue.keys()) == set(demand.keys()) == set(initial_inventory.keys()):
    raise ValueError('Mismatch in product indices among data fields.')
m = gp.Model('MobileSales_NRM')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= initial_inventory[i] for i in product_names), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in product_names), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')