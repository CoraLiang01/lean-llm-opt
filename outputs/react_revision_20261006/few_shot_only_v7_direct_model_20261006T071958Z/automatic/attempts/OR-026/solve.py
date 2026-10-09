import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM17/SupermarketSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except Exception:
            continue
    else:
        raise RuntimeError(f'Could not read CSV at {csv_path} with tried encodings.')
    if 'Product Name' not in df.columns:
        raise ValueError("Missing required column: 'Product Name'")
    fashion_mask = df['Product Name'].str.casefold().str.contains('fashion')
    df_fashion = df[fashion_mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_fashion.columns:
            raise ValueError(f'Missing required column: {col}')
    group_cols = ['Product Name']
    numeric_cols = ['Revenue', 'Demand', 'Initial Inventory']
    for col in numeric_cols:
        df_fashion[col] = pd.to_numeric(df_fashion[col], errors='raise')
    grouped = df_fashion.groupby('Product Name', sort=False, as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = grouped['Product Name'].tolist()
    if not items:
        raise ValueError("No products with 'Fashion' in 'Product Name' found in the data.")
    revenue = dict(zip(grouped['Product Name'], grouped['Revenue']))
    demand = dict(zip(grouped['Product Name'], grouped['Demand']))
    inventory = dict(zip(grouped['Product Name'], grouped['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product: {i}')
    upper_bounds = {i: min(demand[i], inventory[i]) for i in items}
    m = gp.Model('Fashion_Revenue_Maximization')
    quantity_vars = m.addVars(items, lb=0, ub=[upper_bounds[i] for i in items], vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
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