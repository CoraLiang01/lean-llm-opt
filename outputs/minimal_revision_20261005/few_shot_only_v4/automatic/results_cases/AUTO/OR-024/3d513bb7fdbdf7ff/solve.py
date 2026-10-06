import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
    tried_encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in tried_encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with tried encodings.')
    mask = df['Product Name'].astype(str).str.casefold().str.startswith('s700_')
    df_s700 = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_s700.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_s700['Product Name'].astype(str).tolist()
    if len(items) == 0:
        raise ValueError("No products with prefix 'S700_' found in the source data.")
    df_s700_grouped = df_s700.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_s700_grouped['Product Name'].astype(str).tolist()
    revenue = dict(zip(items, df_s700_grouped['Revenue']))
    demand = dict(zip(items, df_s700_grouped['Demand']))
    inventory = dict(zip(items, df_s700_grouped['Initial Inventory']))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing data for product {i}.')
        if pd.isnull(revenue[i]) or pd.isnull(demand[i]) or pd.isnull(inventory[i]):
            raise ValueError(f'Null data for product {i}.')
        if not (isinstance(revenue[i], (int, float)) and isinstance(demand[i], (int, float)) and isinstance(inventory[i], (int, float))):
            raise ValueError(f'Non-numeric data for product {i}.')
    m = gp.Model('S700_Product_Revenue_Max')
    m.Params.MIPGap = 0.0001
    x = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * x[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((x[i] <= demand[i] for i in items), name='')
    m.addConstrs((x[i] <= inventory[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for var in m.getVars():
            print(f'{var.VarName}: {var.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()