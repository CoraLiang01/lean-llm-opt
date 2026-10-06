import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing 'Product Name' column in data.")
    mask = df['Product Name'].str.casefold().str.contains('fashion')
    df_fashion = df[mask].copy()
    if df_fashion.empty:
        raise ValueError("No products classified under 'Fashion' found in the dataset.")
    df_fashion = df_fashion.dropna(subset=['Product Name', 'Revenue', 'Demand', 'Initial Inventory'])
    df_fashion = df_fashion.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_fashion['Product Name'].tolist()
    a = dict(zip(items, df_fashion['Revenue']))
    d = dict(zip(items, df_fashion['Demand']))
    s = dict(zip(items, df_fashion['Initial Inventory']))
    for i in items:
        if pd.isnull(a[i]) or pd.isnull(d[i]) or pd.isnull(s[i]):
            raise ValueError(f'Missing data for product: {i}')
        if not (isinstance(a[i], (int, float)) and isinstance(d[i], (int, float)) and isinstance(s[i], (int, float))):
            raise ValueError(f'Non-numeric data for product: {i}')
    for i in items:
        if d[i] < 0 or s[i] < 0:
            raise ValueError(f'Negative demand or inventory for product: {i}')
        d[i] = int(d[i])
        s[i] = int(s[i])
    m = gp.Model('Fashion_Revenue_Max')
    m.setParam('MIPGap', 0.0001)
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((a[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= d[i] for i in items), name='')
    m.addConstrs((x[i] <= s[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()