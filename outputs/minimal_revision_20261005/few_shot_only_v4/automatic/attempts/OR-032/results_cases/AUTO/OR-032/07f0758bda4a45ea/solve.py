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
    mask = df['Product_Name'].astype(str).str.casefold().str.contains('books')
    books_df = df[mask].copy()
    if books_df.empty:
        raise ValueError("No products classified as 'Books' found in the data.")
    I = list(books_df['Product_Name'].astype(str))

    def build_param_dict(col):
        grouped = books_df.groupby('Product_Name', sort=False)[col].sum()
        return grouped.to_dict()
    a = build_param_dict('Revenue')
    d = build_param_dict('Demand')
    s = build_param_dict('Initial Inventory')
    for i in I:
        if i not in a or i not in d or i not in s:
            raise ValueError(f'Missing coefficients for product: {i}')
    m = gp.Model('Books_Revenue_Maximization')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(I, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((a[i] * x[i] for i in I)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= s[i] for i in I), name='')
    m.addConstrs((x[i] <= d[i] for i in I), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()