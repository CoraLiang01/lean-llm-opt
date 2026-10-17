import gurobipy as gp
from gurobipy import GRB
import pandas as pd
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM16/SmartphoneRetailOutletSalesData.csv'
try:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
except Exception as e:
    raise RuntimeError(f'Failed to read CSV: {e}')
required_columns = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
for col in required_columns:
    if col not in df.columns:
        raise ValueError(f'Missing required column: {col}')
tablet_regex = re.compile('^TABLET_')
tablet_rows = df[df['Product Name'].apply(lambda x: bool(tablet_regex.match(x)))]
if tablet_rows.empty:
    raise ValueError("No products found with Product Name starting with 'TABLET_'.")
items = []
revenue = {}
demand = {}
inventory = {}
for idx, row in tablet_rows.iterrows():
    prod = row['Product Name']
    try:
        rev = float(row['Revenue'])
        dem = int(float(row['Demand']))
        inv = int(float(row['Initial Inventory']))
    except Exception as e:
        raise ValueError(f"Failed to parse numeric fields for product '{prod}': {e}")
    items.append(prod)
    revenue[prod] = rev
    demand[prod] = dem
    inventory[prod] = inv
if not set(items) == set(revenue.keys()) == set(demand.keys()) == set(inventory.keys()):
    raise ValueError('Mismatch in index sets for items, revenue, demand, or inventory.')
m = gp.Model('Tablet_Inventory_Optimization')
m.Params.MIPGap = 0.0001
x_vars = m.addVars(items, vtype=GRB.INTEGER, lb=0, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in x_vars.values():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')