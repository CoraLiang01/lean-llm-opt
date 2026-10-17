import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
import os
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM9/Salesdata.csv'
if not os.path.isfile(csv_path):
    raise FileNotFoundError(f'CSV file not found at {csv_path}')
df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
baby_mask = df['Product Name'].str.contains('\\bBaby\\b', flags=re.IGNORECASE, regex=True)
baby_df = df[baby_mask].copy()
if baby_df.empty:
    raise ValueError("No products classified as 'Baby' found in the dataset.")
items = []
revenue = {}
demand = {}
inventory = {}
for idx, row in baby_df.iterrows():
    prod = row['Product Name']
    for field, dest, dest_dict in [('Revenue', 'A_i', revenue), ('Demand', 'd_i', demand), ('Initial Inventory', 'I_i', inventory)]:
        val_str = row[field]
        m = re.fullmatch('\\s*([+-]?\\d+(?:\\.\\d+)?(?:[eE][+-]?\\d+)?)\\s*', val_str)
        if not m:
            raise ValueError(f"Could not parse numeric value for '{field}' in product '{prod}': '{val_str}'")
        val = float(m.group(1))
        if field in ['Demand', 'Initial Inventory']:
            if not val.is_integer() or val < 0:
                raise ValueError(f"Field '{field}' for product '{prod}' must be a non-negative integer, got '{val_str}'")
            val = int(val)
        dest_dict[prod] = val
    items.append(prod)
for prod in items:
    for dct, name in [(revenue, 'Revenue'), (demand, 'Demand'), (inventory, 'Initial Inventory')]:
        if prod not in dct:
            raise ValueError(f"Missing {name} for product '{prod}'")
m = gp.Model('Baby_Product_Revenue_Max')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')