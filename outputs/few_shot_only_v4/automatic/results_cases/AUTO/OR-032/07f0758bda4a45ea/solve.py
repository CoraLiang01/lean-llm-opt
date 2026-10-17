import pandas as pd
import gurobipy as gp
from gurobipy import GRB
import re
csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'

def parse_csv_books(csv_path):
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    book_rows = []
    for idx, row in df.iterrows():
        pname = row['Product_Name']
        if not isinstance(pname, str):
            continue
        if re.search('\\bBooks?\\b', pname, re.IGNORECASE):
            book_rows.append(idx)
    if not book_rows:
        raise ValueError("No products classified as 'Books' found in Product_Name column.")
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for idx in book_rows:
        pname = df.at[idx, 'Product_Name']
        items.append(pname)
        try:
            rev = float(df.at[idx, 'Revenue'])
        except Exception:
            raise ValueError(f"Unparseable Revenue for product '{pname}': {df.at[idx, 'Revenue']}")
        try:
            dem = float(df.at[idx, 'Demand'])
        except Exception:
            raise ValueError(f"Unparseable Demand for product '{pname}': {df.at[idx, 'Demand']}")
        try:
            inv = float(df.at[idx, 'Initial Inventory'])
        except Exception:
            raise ValueError(f"Unparseable Initial Inventory for product '{pname}': {df.at[idx, 'Initial Inventory']}")
        revenue[pname] = rev
        demand[pname] = dem
        inventory[pname] = inv
    if not set(items) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Mismatch in index sets for items, revenue, demand, or inventory.')
    return (items, revenue, demand, inventory)
items, revenue, demand, inventory = parse_csv_books(csv_path)
m = gp.Model('Books_Revenue_Maximization')
x_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
m.setObjective(gp.quicksum((revenue[i] * x_vars[i] for i in items)), GRB.MAXIMIZE)
m.addConstrs((x_vars[i] <= demand[i] for i in items), name='')
m.addConstrs((x_vars[i] <= inventory[i] for i in items), name='')
m.Params.MIPGap = 0.0001
m.optimize()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for var in m.getVars():
        print(f'{var.VarName}: {var.X}')
else:
    print(f'Solver status: {m.Status}')