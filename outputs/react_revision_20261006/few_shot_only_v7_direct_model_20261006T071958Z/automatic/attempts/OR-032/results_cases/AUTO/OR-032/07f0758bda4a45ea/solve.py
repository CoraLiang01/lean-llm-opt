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
        raise RuntimeError('Could not decode CSV with supported encodings.')
    mask_books = df['Product_Name'].str.casefold().str.startswith('books')
    books_df = df[mask_books].copy()
    required_cols = ['Product_Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in books_df.columns:
            raise ValueError(f'Missing required column: {col}')
    books_df['Revenue'] = pd.to_numeric(books_df['Revenue'], errors='raise')
    books_df['Demand'] = pd.to_numeric(books_df['Demand'], errors='raise')
    books_df['Initial Inventory'] = pd.to_numeric(books_df['Initial Inventory'], errors='raise')
    books_df = books_df.groupby('Product_Name', as_index=False).agg({'Revenue': 'first', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = books_df['Product_Name'].tolist()
    revenue = dict(zip(books_df['Product_Name'], books_df['Revenue']))
    demand = dict(zip(books_df['Product_Name'], books_df['Demand']))
    inventory = dict(zip(books_df['Product_Name'], books_df['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    m = gp.Model('Books_Revenue_Maximization')
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.Params.MIPGap = 0.0001
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()