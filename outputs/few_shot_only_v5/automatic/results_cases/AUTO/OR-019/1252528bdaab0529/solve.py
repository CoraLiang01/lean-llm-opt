import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM10/SalesDataAnalysis.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_cols:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
pattern = re.compile('27in')
mask = df['Product Name'].apply(lambda x: bool(pattern.search(x)))
if not mask.any():
    raise ValueError("No products found with '27in' in 'Product Name'.")
df_27in = df[mask].copy()

def parse_float(val, rowidx, colname):
    try:
        return float(val)
    except Exception:
        raise ValueError(f"Non-numeric value in row {rowidx + 1}, column '{colname}': {val}")

def parse_int(val, rowidx, colname):
    try:
        return int(float(val))
    except Exception:
        raise ValueError(f"Non-integer value in row {rowidx + 1}, column '{colname}': {val}")
items = []
revenue = {}
demand = {}
inventory = {}
for idx, row in df_27in.iterrows():
    pname = row['Product Name']
    items.append(pname)
    revenue[pname] = parse_float(row['Revenue'], idx, 'Revenue')
    demand[pname] = parse_int(row['Demand'], idx, 'Demand')
    inventory[pname] = parse_int(row['Initial Inventory'], idx, 'Initial Inventory')
for pname in items:
    if pname not in revenue or pname not in demand or pname not in inventory:
        raise ValueError(f'Missing data for product: {pname}')
m = gp.Model('NRM_27in_RevMax')
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