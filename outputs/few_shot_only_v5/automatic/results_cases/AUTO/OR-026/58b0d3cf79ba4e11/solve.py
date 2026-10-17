import re
import pandas as pd
import gurobipy as gp
from gurobipy import GRB
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
with open(csv_path, 'r', encoding='utf-8') as f:
    csv_text = f.read()
row_pattern = re.compile('^\\s*(Fashion[^\\n,]*)\\s*,\\s*([0-9.]+)\\s*,\\s*([0-9]+)\\s*,\\s*([0-9]+)\\s*$', re.MULTILINE)
matches = row_pattern.findall(csv_text)
if not matches:
    raise RuntimeError("No 'Fashion' product rows matched in the CSV file.")
fashion_items = []
revenue = {}
demand = {}
inventory = {}
for prod, rev, dem, inv in matches:
    key = prod.strip()
    if key in fashion_items:
        raise ValueError(f'Duplicate product key: {key}')
    fashion_items.append(key)
    try:
        revenue[key] = float(rev)
        demand[key] = int(dem)
        inventory[key] = int(inv)
    except Exception as e:
        raise ValueError(f'Failed to parse numeric fields for {key}: {e}')
if not set(revenue) == set(demand) == set(inventory) == set(fashion_items):
    raise ValueError('Mismatch in index sets for revenue, demand, or inventory.')
m = gp.Model('Fashion_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(fashion_items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in fashion_items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in fashion_items), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in fashion_items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')