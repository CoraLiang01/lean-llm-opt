import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM23/DifferentStoreSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not read CSV with supported encodings.')
    required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    if not df['Product_Name'].dtype == object:
        df['Product_Name'] = df['Product_Name'].astype(str)
    books_mask = df['Product_Name'].str.casefold().str.contains('books')
    books_df = df[books_mask].copy()
    if books_df.empty:
        raise ValueError("No products classified under 'Books' found in the data.")
    items = []
    revenue = {}
    inventory = {}
    demand = {}
    for (idx, row) in books_df.iterrows():
        key = row['Product_Name']
        if pd.isnull(row['Revenue']) or pd.isnull(row['Demand']) or pd.isnull(row['Initial Inventory']):
            raise ValueError(f'Missing data for product: {key}')
        items.append(key)
        revenue[key] = float(row['Revenue'])
        inventory[key] = float(row['Initial Inventory'])
        demand[key] = float(row['Demand'])
    if len(items) != len(set(items)):
        agg_df = books_df.groupby('Product_Name', as_index=False).agg({'Revenue': 'first', 'Initial Inventory': 'sum', 'Demand': 'sum'})
        items = []
        revenue = {}
        inventory = {}
        demand = {}
        for (idx, row) in agg_df.iterrows():
            key = row['Product_Name']
            items.append(key)
            revenue[key] = float(row['Revenue'])
            inventory[key] = float(row['Initial Inventory'])
            demand[key] = float(row['Demand'])
    for key in items:
        if key not in revenue or key not in inventory or key not in demand:
            raise ValueError(f'Missing coefficients for product: {key}')
    m = gp.Model('Books_Revenue_Max')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()