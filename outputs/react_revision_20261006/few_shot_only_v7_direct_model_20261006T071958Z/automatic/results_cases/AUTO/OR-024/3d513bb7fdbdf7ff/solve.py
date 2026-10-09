import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM15/SampleSalesData.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    mask = df['Product Name'].str.casefold().str.startswith('s700_')
    df_s700 = df[mask].copy()
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df_s700.columns:
            raise ValueError(f'Missing required column: {col}')
    items = df_s700['Product Name'].tolist()
    if len(items) == 0:
        raise ValueError("No products found with 'Product Name' starting with 'S700_'.")

    def to_int_series(series, colname):
        try:
            return pd.to_numeric(series, errors='raise').astype(int)
        except Exception:
            raise ValueError(f"Non-numeric or missing value in column '{colname}' for S700_ products.")
    revenue_series = to_int_series(df_s700['Revenue'], 'Revenue')
    demand_series = to_int_series(df_s700['Demand'], 'Demand')
    inventory_series = to_int_series(df_s700['Initial Inventory'], 'Initial Inventory')
    revenue = dict(zip(items, revenue_series))
    demand = dict(zip(items, demand_series))
    inventory = dict(zip(items, inventory_series))
    for i in items:
        if i not in revenue or i not in demand or i not in inventory:
            raise ValueError(f'Missing parameter for product {i}.')
    m = gp.Model('S700_Revenue_Max')
    m.Params.MIPGap = 0.0001
    quantity_vars = m.addVars(items, lb=0, vtype=GRB.INTEGER, name='')
    m.setObjective(gp.quicksum((revenue[i] * quantity_vars[i] for i in items)), GRB.MAXIMIZE)
    m.addConstrs((quantity_vars[i] <= inventory[i] for i in items), name='')
    m.addConstrs((quantity_vars[i] <= demand[i] for i in items), name='')
    m.optimize()
    if m.Status == GRB.OPTIMAL:
        print(f'ObjVal: {m.ObjVal}')
        for v in m.getVars():
            print(f'{v.VarName}: {v.X}')
    else:
        print(f'Solver status: {m.Status}')
    return m
m = solve_problem()