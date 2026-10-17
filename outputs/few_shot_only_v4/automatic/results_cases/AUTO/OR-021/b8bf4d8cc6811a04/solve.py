import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
product_pattern = re.compile('.+')
revenue_pattern = re.compile('^-?\\d+(\\.\\d+)?$')
demand_pattern = re.compile('^-?\\d+$')
inventory_pattern = re.compile('^-?\\d+$')
products = []
revenue = {}
demand = {}
inventory = {}
for idx, row in df.iterrows():
    pname = row['Product Name']
    rev = row['Revenue']
    dem = row['Demand']
    inv = row['Initial Inventory']
    if not product_pattern.fullmatch(pname):
        raise ValueError(f"Unmatched product name at row {idx}: '{pname}'")
    if not revenue_pattern.fullmatch(rev):
        raise ValueError(f"Unmatched revenue at row {idx}: '{rev}'")
    if not demand_pattern.fullmatch(dem):
        raise ValueError(f"Unmatched demand at row {idx}: '{dem}'")
    if not inventory_pattern.fullmatch(inv):
        raise ValueError(f"Unmatched initial inventory at row {idx}: '{inv}'")
    products.append(pname)
    revenue[pname] = float(rev)
    demand[pname] = int(dem)
    inventory[pname] = int(inv)
if not set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()) == set(products):
    raise ValueError('Mismatch in product indices among parameters.')
m = gp.Model('NRM_Summer_Clothes')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(products, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in products)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in products), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in products), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')