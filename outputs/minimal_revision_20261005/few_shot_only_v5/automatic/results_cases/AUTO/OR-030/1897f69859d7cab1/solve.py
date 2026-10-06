import pandas as pd
import gurobipy as gp
from gurobipy import GRB

def solve_problem():
    csv_path = '/Users/cora/Documents/GitHub/lean-llm-opt/Test_Dataset/Large-scale-or/NRM_testing/NRM21/BigMartSales.csv'
    encodings = ['utf-8-sig', 'utf-8', 'gbk', 'latin-1']
    for enc in encodings:
        try:
            df = pd.read_csv(csv_path, encoding=enc)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise RuntimeError('Could not decode CSV with supported encodings.')
    required_cols = ['Product Name', 'Revenue', 'Demand', 'Initial Inventory']
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f'Missing required column: {col}')
    mask = df['Product Name'].astype(str).str.casefold().str.contains('fdk57')
    df_fdk57 = df[mask].copy()
    if df_fdk57.empty:
        raise ValueError("No 'FDK57' car models found in the source data.")
    df_fdk57 = df_fdk57.groupby('Product Name', as_index=False).agg({'Revenue': 'sum', 'Demand': 'sum', 'Initial Inventory': 'sum'})
    items = df_fdk57['Product Name'].tolist()
    revenue = {}
    inventory = {}
    demand = {}
    for (_, row) in df_fdk57.iterrows():
        key = row['Product Name']
        try:
            revenue[key] = float(row['Revenue'])
            inventory[key] = float(row['Initial Inventory'])
            demand[key] = float(row['Demand'])
        except Exception as e:
            raise ValueError(f"Invalid data for product '{key}': {e}")
    for key in items:
        if key not in revenue or key not in inventory or key not in demand:
            raise ValueError(f"Missing coefficients for product '{key}'.")
    m = gp.Model('Car_Dealership_FDK57')
    m.Params.MIPGap = 0.0001
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