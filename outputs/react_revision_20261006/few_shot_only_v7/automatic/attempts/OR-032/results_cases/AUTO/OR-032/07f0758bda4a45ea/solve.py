import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with any of the specified encodings.')
    required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    books_mask = df['Product_Name'].str.casefold().str.contains('books')
    books_df = df[books_mask].copy()
    if books_df.empty:
        raise ValueError("No products classified under 'Books' found in the data.")
    items = []
    revenue = {}
    demand = {}
    inventory = {}
    for (idx, row) in books_df.iterrows():
        prod = row['Product_Name']
        try:
            rev = float(row['Revenue'])
        except Exception:
            raise ValueError(f"Non-numeric Revenue for product '{prod}'")
        try:
            dem = float(row['Demand'])
        except Exception:
            raise ValueError(f"Non-numeric Demand for product '{prod}'")
        try:
            inv = float(row['Initial Inventory'])
        except Exception:
            raise ValueError(f"Non-numeric Initial Inventory for product '{prod}'")
        items.append(prod)
        revenue[prod] = rev
        demand[prod] = dem
        inventory[prod] = inv
    if not set(items) == set(revenue) == set(demand) == set(inventory):
        raise ValueError('Mismatch in coefficient coverage for items.')
    m = gp.Model('Books_Revenue_Maximization')
    quantity_vars = m.addVars(items, lb=0, ub={i: min(demand[i], inventory[i]) for i in items}, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    return m
m = solve_problem()
if m.Status == GRB.OPTIMAL:
    print(f'ObjVal: {m.ObjVal}')
    for v in m.getVars():
        print(f'{v.VarName}: {v.X}')
else:
    print(f'Solver status: {m.Status}')