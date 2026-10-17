import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM12/Salesofsummerclothes.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
product_names = []
revenue = {}
demand = {}
inventory = {}
row_pattern = re.compile('.+')
for idx, row in df.iterrows():
    pname = row['Product Name']
    if not row_pattern.match(pname):
        raise ValueError(f'Unmatched Product Name in row {idx}: {pname}')
    product_names.append(pname)
    rev_str = row['Revenue']
    if not re.match('^-?\\d+(\\.\\d+)?$', rev_str):
        raise ValueError(f'Unmatched Revenue in row {idx}: {rev_str}')
    revenue[pname] = float(rev_str)
    d_str = row['Demand']
    if not re.match('^\\d+$', d_str):
        raise ValueError(f'Unmatched Demand in row {idx}: {d_str}')
    demand[pname] = int(d_str)
    inv_str = row['Initial Inventory']
    if not re.match('^\\d+$', inv_str):
        raise ValueError(f'Unmatched Initial Inventory in row {idx}: {inv_str}')
    inventory[pname] = int(inv_str)
if set(product_names) != set(revenue.keys()) or set(product_names) != set(demand.keys()) or set(product_names) != set(inventory.keys()):
    raise ValueError('Mismatch in index set coverage for products, revenue, demand, or inventory.')
m = gp.Model('NRM_SummerClothes')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(product_names, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in product_names)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in product_names), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in product_names), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')