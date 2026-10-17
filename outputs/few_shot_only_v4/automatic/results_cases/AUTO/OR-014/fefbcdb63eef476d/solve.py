import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM5/PizzaSalesDataset.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
with open(csv_path, 'r', encoding='utf-8') as f:
    header = f.readline()
    header_cols = [h.strip() for h in header.strip().split(',')]
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in header_cols:
            raise ValueError(f"Required column '{col}' not found in CSV header.")
    col_idx = {col: header_cols.index(col) for col in required_cols}
    pattern = re.compile('^(.*?),(.*?),(.*?),(.*?)$')
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for line in f:
        line = line.strip()
        if not line:
            continue
        mobj = pattern.match(line)
        if not mobj:
            raise ValueError(f'CSV row does not match expected format: {line}')
        fields = [mobj.group(i + 1).strip() for i in range(4)]
        pname = fields[col_idx['Product Name']]
        try:
            rev = float(fields[col_idx['Revenue']])
            dem = int(float(fields[col_idx['Demand']]))
            inv = int(float(fields[col_idx['Initial Inventory']]))
        except Exception as e:
            raise ValueError(f'Failed to parse numeric fields in row: {line}\nError: {e}')
        if pname in items:
            raise ValueError(f'Duplicate product name found: {pname}')
        items.append(pname)
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
if not set(revenue) == set(demand) == set(inventory) == set(items):
    raise ValueError('Mismatch in index sets for revenue, demand, or inventory.')
m = gp.Model('Pizza_Fulfillment')
x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')