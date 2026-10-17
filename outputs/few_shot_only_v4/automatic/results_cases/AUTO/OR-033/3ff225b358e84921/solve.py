import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM24/EuropeSalesRecords.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV at {csv_path}: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f"Required column '{col}' not found in CSV.")
baby_mask = df['Product Name'].str.contains('\\bBaby\\b', flags=re.IGNORECASE, regex=True)
if not baby_mask.any():
    raise ValueError("No products classified under 'Baby' found in 'Product Name' column.")
baby_df = df.loc[baby_mask].copy()
if baby_df.empty:
    raise ValueError("No rows found for 'Baby' products after filtering.")
items = []
revenue = {}
demand = {}
inventory = {}
for idx, row in baby_df.iterrows():
    prod = row['Product Name']
    revenue_match = re.fullmatch('\\s*(-?\\d+(?:\\.\\d+)?)\\s*', row['Revenue'])
    demand_match = re.fullmatch('\\s*(-?\\d+)\\s*', row['Demand'])
    inventory_match = re.fullmatch('\\s*(-?\\d+)\\s*', row['Initial Inventory'])
    if not revenue_match:
        raise ValueError(f"Unmatched or invalid Revenue for product '{prod}': '{row['Revenue']}'")
    if not demand_match:
        raise ValueError(f"Unmatched or invalid Demand for product '{prod}': '{row['Demand']}'")
    if not inventory_match:
        raise ValueError(f"Unmatched or invalid Initial Inventory for product '{prod}': '{row['Initial Inventory']}'")
    rev = float(revenue_match.group(1))
    dem = int(demand_match.group(1))
    inv = int(inventory_match.group(1))
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
if not set(items) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Mismatch in parameter keys for items, revenue, demand, or inventory.')

def build_baby_revenue_model(items, revenue, demand, inventory):
    m = gp.Model('Baby_Product_Revenue')
    x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    return m
m = build_baby_revenue_model(items, revenue, demand, inventory)
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')